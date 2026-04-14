/**
 * MetricsChart — 학습 loss/acc 실시간 라인차트 (Recharts)
 * Phase 1/2 구분선 포함
 */
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrainMetric } from "../types";

interface Props {
  metrics:       TrainMetric[];
  phase1Epochs?: number;
}

const fmt2 = (v: number) => v.toFixed(4);

export default function MetricsChart({ metrics, phase1Epochs }: Props) {
  if (metrics.length === 0) {
    return (
      <div className="flex items-center justify-center h-52 text-gray-600 text-sm">
        학습 시작 후 차트가 표시됩니다
      </div>
    );
  }

  // x축: 누적 epoch 번호
  const data = metrics.map((m, i) => ({
    x:          i + 1,
    train_loss: m.train_loss,
    val_loss:   m.val_loss,
    train_acc:  m.train_acc,
    val_acc:    m.val_acc,
    phase:      m.phase,
  }));

  // Phase 2 시작 지점
  const phase2Start = phase1Epochs
    ? phase1Epochs + 0.5
    : data.findIndex((d) => d.phase === 2) + 0.5;

  return (
    <div className="space-y-4">
      {/* Loss */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Loss</p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="x" tick={{ fontSize: 10, fill: "#6b7280" }} />
            <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} tickFormatter={fmt2} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 11 }}
              formatter={fmt2}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {phase2Start > 0 && (
              <ReferenceLine x={phase2Start} stroke="#6d8fff" strokeDasharray="4 4" label={{ value: "Phase 2", fill: "#6d8fff", fontSize: 10 }} />
            )}
            <Line type="monotone" dataKey="train_loss" stroke="#f59e0b" dot={false} name="train" strokeWidth={1.5} />
            <Line type="monotone" dataKey="val_loss"   stroke="#ef4444" dot={false} name="val"   strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Accuracy</p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="x" tick={{ fontSize: 10, fill: "#6b7280" }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#6b7280" }} tickFormatter={fmt2} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 11 }}
              formatter={fmt2}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {phase2Start > 0 && (
              <ReferenceLine x={phase2Start} stroke="#6d8fff" strokeDasharray="4 4" />
            )}
            <Line type="monotone" dataKey="train_acc" stroke="#34d399" dot={false} name="train" strokeWidth={1.5} />
            <Line type="monotone" dataKey="val_acc"   stroke="#60a5fa" dot={false} name="val"   strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
