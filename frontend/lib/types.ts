export type BoundingBoxData = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type Detection = {
  id: string;
  label: string;
  confidence: number;
  bbox: BoundingBoxData;
};

export type DetectionResponse = {
  inferenceTime: number;
  detections: Detection[];
  resultImageUrl?: string;
  model: string;
};

export type RequestState = "idle" | "loading" | "success" | "error";
