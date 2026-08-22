import { Scan } from "@phosphor-icons/react/dist/ssr";
import type { DetectionResponse, RequestState } from "@/lib/types";
import ImagePreview from "./ImagePreview";

type SourceImage = {
  url?: string;
  name?: string;
  isTiff?: boolean;
};

type ResultViewerProps = {
  rgb: SourceImage;
  sar: SourceImage;
  result: DetectionResponse | null;
  state: RequestState;
};

function ViewerPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <article className="viewer-panel">
      <div className="viewer-panel__header">
        <h3>{title}</h3>
      </div>
      <div className="viewer-panel__canvas">{children}</div>
    </article>
  );
}

export default function ResultViewer({ rgb, sar, result, state }: ResultViewerProps) {
  const isLoading = state === "loading";
  const rgbResultSource = result?.rgbResultImageUrl ?? result?.resultImageUrl ?? rgb.url;
  const sarResultSource = result?.sarResultImageUrl ?? sar.url;

  const renderResult = (
    source: string | undefined,
    fallback: SourceImage,
    alt: string,
    hasRenderedImage: boolean,
  ) => {
    if (isLoading) {
      return (
        <div className="result-loading">
          <div className="scan-line" />
          <Scan size={30} weight="thin" aria-hidden="true" />
          <span>Processing multimodal inputs</span>
        </div>
      );
    }

    return (
      <ImagePreview
        src={result ? source : undefined}
        alt={alt}
        fileName={fallback.name}
        isTiff={Boolean(result && !hasRenderedImage && fallback.isTiff)}
        detections={result && !hasRenderedImage ? result.detections : []}
        emptyLabel="Run detection to generate output"
      />
    );
  };

  return (
    <section className="results-section" aria-labelledby="results-title" aria-busy={isLoading}>
      <div className="section-heading">
        <div>
          <span>Output Viewer</span>
          <h2 id="results-title">Detection Results</h2>
        </div>
        {result ? <strong>{result.detections.length} objects detected</strong> : null}
      </div>

      <div className="viewer-grid">
        <ViewerPanel title="RGB Detection Result">
          {renderResult(
            rgbResultSource,
            rgb,
            "YOLO-CMFM RGB detection result",
            Boolean(result?.rgbResultImageUrl ?? result?.resultImageUrl),
          )}
        </ViewerPanel>

        <ViewerPanel title="SAR Detection Result">
          {renderResult(
            sarResultSource,
            sar,
            "YOLO-CMFM SAR detection result",
            Boolean(result?.sarResultImageUrl),
          )}
        </ViewerPanel>
      </div>

      {result ? (
        <div className="detection-list" aria-label="Detected objects">
          {result.detections.map((detection) => (
            <div key={detection.id}>
              <span>{detection.label}</span>
              <strong>{detection.confidence.toFixed(2)}</strong>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
