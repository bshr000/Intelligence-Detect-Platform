"use client";

import { useEffect, useState } from "react";
import { Cpu, Database, Images, Timer } from "@phosphor-icons/react";
import DetectionPanel from "@/components/DetectionPanel";
import ImageUploader from "@/components/ImageUploader";
import InfoCard from "@/components/InfoCard";
import ResultViewer from "@/components/ResultViewer";
import { detect } from "@/lib/api";
import type { DetectionResponse, RequestState } from "@/lib/types";

function isTiff(file: File | null) {
  if (!file) return false;
  const extension = file.name.split(".").pop()?.toLowerCase();
  return extension === "tif" || extension === "tiff";
}

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string>();

  useEffect(() => {
    if (!file || isTiff(file)) {
      setUrl(undefined);
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  return url;
}

export default function DetectionPlatform() {
  const [rgbFile, setRgbFile] = useState<File | null>(null);
  const [sarFile, setSarFile] = useState<File | null>(null);
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string>();
  const rgbUrl = useObjectUrl(rgbFile);
  const sarUrl = useObjectUrl(sarFile);

  const resetOutput = () => {
    setResult(null);
    setError(undefined);
    setRequestState("idle");
  };

  const handleRgbChange = (file: File | null) => {
    setRgbFile(file);
    resetOutput();
  };

  const handleSarChange = (file: File | null) => {
    setSarFile(file);
    resetOutput();
  };

  const runDetection = async () => {
    if (!rgbFile || !sarFile) return;
    setRequestState("loading");
    setError(undefined);
    setResult(null);

    try {
      const nextResult = await detect(rgbFile, sarFile);
      setResult(nextResult);
      setRequestState("success");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unknown request error");
      setRequestState("error");
    }
  };

  const busy = requestState === "loading";

  return (
    <main className="platform-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            CM
          </div>
          <div>
            <h1>YOLO-CMFM Multimodal Detection</h1>
            <p>Visible-SAR Remote Sensing Object Detection System</p>
          </div>
        </div>
        <div className="system-status" role="status">
          <span aria-hidden="true" />
          Interface ready
        </div>
      </header>

      <div className="workspace">
        <section className="input-section" aria-labelledby="input-title">
          <div className="section-heading section-heading--compact">
            <div>
              <span>Multimodal Input</span>
              <h2 id="input-title">Image Pair</h2>
            </div>
            <p>Upload spatially aligned visible and SAR imagery.</p>
          </div>

          <div className="upload-grid">
            <ImageUploader
              label="RGB Image"
              description="Visible spectrum source"
              file={rgbFile}
              onChange={handleRgbChange}
              disabled={busy}
            />
            <ImageUploader
              label="SAR Image"
              description="Synthetic aperture radar source"
              file={sarFile}
              onChange={handleSarChange}
              disabled={busy}
            />
          </div>
        </section>

        <DetectionPanel
          state={requestState}
          canRun={Boolean(rgbFile && sarFile)}
          error={error}
          onRun={runDetection}
        />

        <ResultViewer
          rgb={{ url: rgbUrl, name: rgbFile?.name, isTiff: isTiff(rgbFile) }}
          sar={{ url: sarUrl, name: sarFile?.name, isTiff: isTiff(sarFile) }}
          result={result}
          state={requestState}
        />

        <section className="metrics-section" aria-labelledby="metrics-title">
          <div className="section-heading section-heading--compact">
            <div>
              <span>Runtime Telemetry</span>
              <h2 id="metrics-title">Detection Information</h2>
            </div>
          </div>
          <div className="metrics-grid">
            <div className="metric-icon" aria-hidden="true"><Timer size={20} /></div>
            <InfoCard
              label="Inference Time"
              value={result ? result.inferenceTime.toFixed(2) : "--"}
              unit="s"
            />
            <div className="metric-icon" aria-hidden="true"><Images size={20} /></div>
            <InfoCard label="Objects" value={result ? String(result.detections.length) : "--"} />
            <div className="metric-icon" aria-hidden="true"><Cpu size={20} /></div>
            <InfoCard label="Model" value={result?.model ?? "YOLO-CMFM"} />
            <div className="metric-icon" aria-hidden="true"><Database size={20} /></div>
            <InfoCard label="Weights" value="best.pt" />
          </div>
        </section>
      </div>
    </main>
  );
}
