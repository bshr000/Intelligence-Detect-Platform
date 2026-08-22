"use client";

import { Play, SpinnerGap, WarningCircle } from "@phosphor-icons/react";
import type { RequestState } from "@/lib/types";

type DetectionPanelProps = {
  state: RequestState;
  canRun: boolean;
  error?: string;
  onRun: () => void;
};

const specs = [
  ["Model", "YOLO-CMFM"],
  ["Weights", "best.pt"],
  ["Input", "RGB + SAR"],
  ["Endpoint", "POST /api/v1/detections"],
];

export default function DetectionPanel({ state, canRun, error, onRun }: DetectionPanelProps) {
  const isLoading = state === "loading";

  return (
    <section className="control-panel" aria-labelledby="detection-control-title">
      <div className="control-panel__title">
        <span>Inference Control</span>
        <h2 id="detection-control-title">Detection Job</h2>
      </div>

      <dl className="model-specs">
        {specs.map(([term, value]) => (
          <div key={term}>
            <dt>{term}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>

      <div className="run-control">
        <button
          type="button"
          className="run-button"
          onClick={onRun}
          disabled={!canRun || isLoading}
        >
          {isLoading ? (
            <SpinnerGap className="spin" size={18} aria-hidden="true" />
          ) : (
            <Play size={18} weight="fill" aria-hidden="true" />
          )}
          {isLoading ? "Running Detection" : "Run Detection"}
        </button>
        <p>{canRun ? "Inputs validated. Ready to submit." : "Upload both input images to continue."}</p>
      </div>

      {error ? (
        <div className="request-error" role="alert">
          <WarningCircle size={18} aria-hidden="true" />
          <div>
            <strong>Detection failed</strong>
            <p>{error}</p>
          </div>
        </div>
      ) : null}
    </section>
  );
}
