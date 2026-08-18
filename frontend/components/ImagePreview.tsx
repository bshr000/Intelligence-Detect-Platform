import { FileImage } from "@phosphor-icons/react/dist/ssr";
import type { Detection } from "@/lib/types";
import BoundingBox from "./BoundingBox";

type ImagePreviewProps = {
  src?: string;
  alt: string;
  fileName?: string;
  detections?: Detection[];
  isTiff?: boolean;
  emptyLabel?: string;
};

export default function ImagePreview({
  src,
  alt,
  fileName,
  detections = [],
  isTiff = false,
  emptyLabel = "No image loaded",
}: ImagePreviewProps) {
  if (isTiff) {
    return (
      <div className="preview-placeholder" role="img" aria-label={`${alt}: ${fileName ?? "TIFF image"}`}>
        <FileImage size={28} weight="thin" aria-hidden="true" />
        <span>TIFF ready for inference</span>
        <small>{fileName}</small>
        <small>Browser preview unavailable</small>
      </div>
    );
  }

  if (!src) {
    return (
      <div className="preview-placeholder">
        <FileImage size={28} weight="thin" aria-hidden="true" />
        <span>{emptyLabel}</span>
      </div>
    );
  }

  return (
    <div className="image-stage">
      {/* Blob URLs and FastAPI response URLs are not known at build time. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className="image-stage__image" />
      {detections.map((detection) => (
        <BoundingBox key={detection.id} detection={detection} />
      ))}
    </div>
  );
}
