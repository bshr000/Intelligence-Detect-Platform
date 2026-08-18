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
  const resultSource = result?.resultImageUrl ?? rgb.url;

  return (
    <section className="results-section" aria-labelledby="results-title" aria-busy={isLoading}>
      <div className="section-heading">
        <div>
          <span>Output Viewer</span>
          <h2 id="results-title">Detection Result</h2>
        </div>
        {result ? <strong>{result.detections.length} objects detected</strong> : null}
      </div>

      <div className="viewer-grid">
        <ViewerPanel title="RGB Input">
          <ImagePreview
            src={rgb.url}
            alt="RGB input image"
            fileName={rgb.name}
            isTiff={rgb.isTiff}
          />
        </ViewerPanel>

        <ViewerPanel title="SAR Input">
          <ImagePreview
            src={sar.url}
            alt="SAR input image"
            fileName={sar.name}
            isTiff={sar.isTiff}
          />
        </ViewerPanel>

        <ViewerPanel title="Detection Result">
          {isLoading ? (
            <div className="result-loading">
              <div className="scan-line" />
              <Scan size={30} weight="thin" aria-hidden="true" />
              <span>Processing multimodal inputs</span>
            </div>
          ) : (
            <ImagePreview
              src={result ? resultSource : undefined}
              alt="YOLO-CMFM detection result"
              fileName={rgb.name}
              isTiff={Boolean(result && !result.resultImageUrl && rgb.isTiff)}
              detections={result?.resultImageUrl ? [] : result?.detections}
              emptyLabel="Run detection to generate output"
            />
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
