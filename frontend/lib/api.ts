import type { Detection, DetectionResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
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
  model?: string;
};

function resolveApiUrl(path?: string) {
  if (!path) return undefined;
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
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

  const response = await fetch(`${API_BASE_URL}/api/v1/detections`, {
    method: "POST",
    body: formData,
  });

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
