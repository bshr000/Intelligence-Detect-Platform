import type { Detection, DetectionResponse } from "./types";

// Empty by default: the browser calls the same Next.js origin and Next proxies
// /api/* to FastAPI. Set NEXT_PUBLIC_API_BASE_URL only for direct API access.
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");
const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API === "true";

type BackendDetection = {
  id?: string;
  label?: string;
  class_name?: string;
  confidence: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

type BackendResponse = {
  inference_time?: number;
  inferenceTime?: number;
  detections: BackendDetection[];
  result_image_url?: string;
  resultImageUrl?: string;
  rgb_result_image_url?: string;
  rgbResultImageUrl?: string;
  sar_result_image_url?: string;
  sarResultImageUrl?: string;
  model?: string;
};

function resolveApiUrl(path?: string) {
  if (!path) return undefined;
  if (/^https?:\/\//i.test(path)) return path;
  const normalizedPath = `/${path.replace(/^\//, "")}`;
  return API_BASE_URL ? `${API_BASE_URL}${normalizedPath}` : normalizedPath;
}

const mockDetections: Detection[] = [
  {
    id: "bridge-01",
    label: "Bridge",
    confidence: 0.93,
    bbox: { x: 0.12, y: 0.18, width: 0.31, height: 0.2 },
  },
  {
    id: "harbor-01",
    label: "Harbor",
    confidence: 0.88,
    bbox: { x: 0.56, y: 0.12, width: 0.3, height: 0.27 },
  },
  {
    id: "tank-01",
    label: "Oil Tank",
    confidence: 0.91,
    bbox: { x: 0.47, y: 0.59, width: 0.22, height: 0.25 },
  },
];

function normalizeResponse(data: BackendResponse): DetectionResponse {
  return {
    inferenceTime: data.inference_time ?? data.inferenceTime ?? 0,
    resultImageUrl: resolveApiUrl(data.result_image_url ?? data.resultImageUrl),
    rgbResultImageUrl: resolveApiUrl(
      data.rgb_result_image_url ?? data.rgbResultImageUrl ?? data.result_image_url ?? data.resultImageUrl,
    ),
    sarResultImageUrl: resolveApiUrl(data.sar_result_image_url ?? data.sarResultImageUrl),
    model: data.model ?? "YOLO-CMFM",
    detections: data.detections.map((item, index) => ({
      id: item.id ?? `detection-${index}`,
      label: item.label ?? item.class_name ?? "Object",
      confidence: item.confidence,
      bbox: item.bbox,
    })),
  };
}

export async function detect(rgbFile: File, sarFile: File): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("rgb_image", rgbFile);
  formData.append("sar_image", sarFile);

  if (USE_MOCK_API) {
    await new Promise((resolve) => setTimeout(resolve, 900));
    return {
      inferenceTime: 0.25,
      detections: mockDetections,
      model: "YOLO-CMFM",
    };
  }

  const endpoint = resolveApiUrl("/api/v1/detections");
  let response: Response;
  try {
    response = await fetch(endpoint!, {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    const reason = error instanceof Error ? ` (${error.message})` : "";
    throw new Error(
      `Cannot reach the inference API at ${endpoint}. Check that FastAPI is running and the API proxy is configured${reason}`,
    );
  }

  if (!response.ok) {
    const body = await response.text();
    let message = body;
    try {
      const parsed = JSON.parse(body) as { detail?: string };
      message = parsed.detail ?? body;
    } catch {
      // Keep the raw response when it is not JSON.
    }
    throw new Error(message || `Detection request failed with status ${response.status}`);
  }

  const data = (await response.json()) as BackendResponse;
  return normalizeResponse(data);
}
