import type { Detection } from "@/lib/types";

type BoundingBoxProps = {
  detection: Detection;
};

export default function BoundingBox({ detection }: BoundingBoxProps) {
  const { bbox, label, confidence } = detection;

  return (
    <div
      className="bounding-box"
      style={{
        left: `${bbox.x * 100}%`,
        top: `${bbox.y * 100}%`,
        width: `${bbox.width * 100}%`,
        height: `${bbox.height * 100}%`,
      }}
      aria-label={`${label}, confidence ${Math.round(confidence * 100)} percent`}
    >
      <span className="bounding-box__label">
        {label} <strong>{confidence.toFixed(2)}</strong>
      </span>
    </div>
  );
}
