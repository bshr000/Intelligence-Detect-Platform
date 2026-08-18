"use client";

import { useEffect, useId, useRef, useState } from "react";
import { FileArrowUp, Trash, WarningCircle } from "@phosphor-icons/react";
import ImagePreview from "./ImagePreview";

const MAX_FILE_SIZE = 50 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "tif", "tiff"];

type ImageUploaderProps = {
  label: "RGB Image" | "SAR Image";
  description: string;
  file: File | null;
  onChange: (file: File | null) => void;
  disabled?: boolean;
};

function getExtension(name: string) {
  return name.split(".").pop()?.toLowerCase() ?? "";
}

function validateFile(file: File) {
  if (!SUPPORTED_EXTENSIONS.includes(getExtension(file.name))) {
    return "Unsupported format. Select PNG, JPG, or TIFF.";
  }
  if (file.size > MAX_FILE_SIZE) {
    return "File exceeds the 50 MB upload limit.";
  }
  return null;
}

export default function ImageUploader({
  label,
  description,
  file,
  onChange,
  disabled = false,
}: ImageUploaderProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>();
  const isTiff = file ? ["tif", "tiff"].includes(getExtension(file.name)) : false;

  useEffect(() => {
    if (!file || isTiff) {
      setPreviewUrl(undefined);
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file, isTiff]);

  const selectFile = (candidate?: File) => {
    if (!candidate) return;
    const validationError = validateFile(candidate);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    onChange(candidate);
  };

  const removeFile = () => {
    setError(null);
    onChange(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <section className="upload-panel" aria-labelledby={`${inputId}-label`}>
      <div className="panel-heading">
        <div>
          <h2 id={`${inputId}-label`}>{label}</h2>
          <p>{description}</p>
        </div>
        <span className="format-label">PNG / JPG / TIFF</span>
      </div>

      <div
        className={`drop-zone ${isDragging ? "drop-zone--active" : ""} ${file ? "drop-zone--loaded" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          if (!disabled) setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          if (!disabled) selectFile(event.dataTransfer.files[0]);
        }}
      >
        {file ? (
          <div className="loaded-image">
            <div className="loaded-image__preview">
              <ImagePreview
                src={previewUrl}
                alt={`${label} preview`}
                fileName={file.name}
                isTiff={isTiff}
              />
            </div>
            <div className="file-row">
              <div className="file-row__details">
                <strong>{file.name}</strong>
                <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
              <button type="button" className="icon-button" onClick={removeFile} disabled={disabled}>
                <Trash size={17} aria-hidden="true" />
                <span className="sr-only">Remove {label}</span>
              </button>
            </div>
          </div>
        ) : (
          <button
            type="button"
            className="drop-zone__prompt"
            onClick={() => inputRef.current?.click()}
            disabled={disabled}
          >
            <FileArrowUp size={30} weight="thin" aria-hidden="true" />
            <span>Drop {label.replace(" Image", "")} image here</span>
            <small>or click to browse local files</small>
          </button>
        )}
      </div>

      <input
        id={inputId}
        ref={inputRef}
        className="sr-only"
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
        onChange={(event) => selectFile(event.target.files?.[0])}
        disabled={disabled}
      />

      {error ? (
        <p className="field-error" role="alert">
          <WarningCircle size={15} aria-hidden="true" />
          {error}
        </p>
      ) : null}
    </section>
  );
}
