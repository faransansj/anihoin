import type { JobState } from "../types";

const styles: Record<JobState, string> = {
  idle:    "bg-gray-700 text-gray-300",
  running: "bg-blue-900 text-blue-300 animate-pulse",
  done:    "bg-green-900 text-green-300",
  failed:  "bg-red-900 text-red-300",
};

const labels: Record<JobState, string> = {
  idle:    "대기",
  running: "실행 중",
  done:    "완료",
  failed:  "실패",
};

export default function StatusBadge({ state }: { state: JobState }) {
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${styles[state]}`}>
      {labels[state]}
    </span>
  );
}
