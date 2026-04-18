/**
 * JobConsole — WebSocket 로그 스트리밍 콘솔
 * jobPath: "/crawl/logs" | "/training/logs" | "/export/logs/fp16" 등
 * onState: state 변경 콜백
 * onMetric: metric 이벤트 콜백 (학습 전용)
 */
import { useEffect, useRef, useState } from "react";
import type { JobState, TrainMetric, TrainProgress } from "../types";
import { api } from "../api";

interface Props {
  jobPath:    string;
  onState?:   (s: JobState) => void;
  onMetric?:  (m: TrainMetric) => void;
  onProgress?: (p: TrainProgress) => void;
  maxLines?:  number;
}

export default function JobConsole({
  jobPath,
  onState,
  onMetric,
  onProgress,
  maxLines = 400,
}: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const wsRef     = useRef<WebSocket | null>(null);

  useEffect(() => {
    let ws: WebSocket;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let dead = false;

    function connect() {
      if (dead) return;
      ws = api.ws(jobPath);
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string);
          if (msg.type === "log") {
            setLines((prev) => {
              const next = [...prev, msg.data as string];
              return next.length > maxLines ? next.slice(-maxLines) : next;
            });
          } else if (msg.type === "state" && onState) {
            onState(msg.data as JobState);
          } else if (msg.type === "metric" && onMetric) {
            onMetric(msg.data as TrainMetric);
          } else if (msg.type === "progress" && onProgress) {
            onProgress(msg.data as TrainProgress);
          }
        } catch {
          /* 무시 */
        }
      };

      ws.onclose = () => {
        if (!dead) retryTimer = setTimeout(connect, 2000);
      };
    }

    setLines([]);
    connect();

    return () => {
      dead = true;
      if (retryTimer) clearTimeout(retryTimer);
      ws?.close();
    };
  }, [jobPath]); // eslint-disable-line react-hooks/exhaustive-deps

  // 자동 스크롤
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  return (
    <div className="bg-gray-950 border border-gray-800 rounded-lg h-64 overflow-y-auto p-3 font-mono text-xs text-gray-300">
      {lines.length === 0 && (
        <span className="text-gray-600 italic">로그 대기 중...</span>
      )}
      {lines.map((line, i) => (
        <div key={i} className="leading-5 whitespace-pre-wrap break-all">
          {line}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
