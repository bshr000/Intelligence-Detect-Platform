type InfoCardProps = {
  label: string;
  value: string;
  unit?: string;
};

export default function InfoCard({ label, value, unit }: InfoCardProps) {
  return (
    <div className="info-card">
      <span>{label}</span>
      <strong>
        {value}
        {unit ? <small>{unit}</small> : null}
      </strong>
    </div>
  );
}
