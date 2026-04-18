/**
 * Export 페이지
 * - FP32 / FP16 / ONNX 모델 카드
 * - 변환 실행 + 로그
 * - 다운로드
 */
import { useEffect, useState } from "react";
import { api } from "../api";
import JobConsole from "../components/JobConsole";
import StatusBadge from "../components/StatusBadge";
import { useJobStore } from "../store/jobStore";
import type { ExportStatus, ModelMap } from "../types";

export default function Export() {
  const { fp16State, onnxState, setFp16State, setOnnxState } = useJobStore();
  const [models, setModels] = useState<ModelMap | null>(null);
  const [opset,  setOpset]  = useState(18);
  const [logTab, setLogTab] = useState<"fp16" | "onnx">("fp16");

  async function loadModels() {
    const r = await api.get<{ models: ModelMap }>("/export/models");
    setModels(r.models);
  }

  useEffect(() => {
    loadModels();
    api.get<ExportStatus>("/export/status")
      .then((r) => { setFp16State(r.fp16.state); setOnnxState(r.onnx.state); })
      .catch(console.error);
  }, []);

  // 변환 완료 시 모델 목록 갱신
  useEffect(() => {
    if (fp16State === "done" || onnxState === "done") loadModels();
  }, [fp16State, onnxState]);

  async function startFp16() {
    await api.post("/export/fp16");
  }
  async function startOnnx() {
    await api.post("/export/onnx", { opset });
  }

  const modelCards = [
    {
      key:   "fp32" as const,
      label: "FP32 (원본)",
      color: "text-gray-300",
      canExport: false,
    },
    {
      key:   "fp16" as const,
      label: "FP16 (경량)",
      color: "text-yellow-300",
      canExport: true,
      running: fp16State === "running",
      onStart: startFp16,
      onStop:  () => api.post("/export/fp16/stop"),
      state: fp16State,
    },
    {
      key:   "onnx" as const,
      label: "ONNX (배포)",
      color: "text-blue-300",
      canExport: true,
      running: onnxState === "running",
      onStart: startOnnx,
      onStop:  () => api.post("/export/onnx/stop"),
      state: onnxState,
    },
  ];

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-bold text-white">Export</h1>
        <p className="text-sm text-gray-400 mt-0.5">모델 양자화 및 배포 형식 변환</p>
      </div>

      {/* ── 모델 카드 ── */}
      <div className="grid grid-cols-3 gap-4">
        {modelCards.map((card) => {
          const entry = models?.[card.key];
          return (
            <div key={card.key} className="card space-y-3">
              <div className="flex items-center justify-between">
                <span className={`font-semibold text-sm ${card.color}`}>{card.label}</span>
                {card.state && <StatusBadge state={card.state} />}
              </div>

              <div className="space-y-1 text-xs text-gray-400">
                <div className="flex justify-between">
                  <span>상태</span>
                  <span className={entry?.exists ? "text-green-400" : "text-gray-600"}>
                    {entry?.exists ? "✓ 존재" : "없음"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>크기</span>
                  <span>{entry?.size_mb != null ? `${entry.size_mb} MB` : "—"}</span>
                </div>
              </div>

              {card.canExport && (
                <div className="space-y-2">
                  {card.key === "onnx" && (
                    <div>
                      <label className="label-text">Opset</label>
                      <input type="number" value={opset} min={11} max={20}
                        onChange={(e) => setOpset(+e.target.value)}
                        className="input text-xs py-1" disabled={card.running} />
                    </div>
                  )}
                  {!card.running ? (
                    <button
                      onClick={card.onStart}
                      disabled={!models?.fp32.exists}
                      className="btn-primary w-full text-xs"
                    >
                      변환 시작
                    </button>
                  ) : (
                    <button onClick={card.onStop} className="btn-danger w-full text-xs">
                      중단
                    </button>
                  )}
                </div>
              )}

              {entry?.exists && (
                <a
                  href={`/api/export/download/${entry.filename}`}
                  download
                  className="block text-center btn-ghost text-xs py-1"
                >
                  ⬇ 다운로드
                </a>
              )}
            </div>
          );
        })}
      </div>

      {/* ── 비교 테이블 ── */}
      {models && (models.fp16.exists || models.onnx.exists) && (
        <div className="card">
          <p className="text-sm font-medium text-gray-200 mb-3">모델 비교</p>
          <table className="w-full text-xs text-gray-300">
            <thead>
              <tr className="text-gray-500 border-b border-gray-700">
                <th className="text-left pb-2">형식</th>
                <th className="text-right pb-2">크기</th>
                <th className="text-right pb-2">압축률</th>
              </tr>
            </thead>
            <tbody>
              {(["fp32", "fp16", "onnx"] as const).map((k) => {
                const e = models[k];
                if (!e.exists) return null;
                const ratio = models.fp32.size_mb && e.size_mb
                  ? Math.round((e.size_mb / models.fp32.size_mb) * 100)
                  : 100;
                return (
                  <tr key={k} className="border-b border-gray-800">
                    <td className="py-1.5 font-medium">{k.toUpperCase()}</td>
                    <td className="text-right">{e.size_mb} MB</td>
                    <td className="text-right text-gray-500">{ratio}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ── 로그 ── */}
      <div className="card">
        <div className="flex gap-2 mb-2">
          {(["fp16", "onnx"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setLogTab(t)}
              className={`text-xs px-3 py-1 rounded-md transition-colors ${
                logTab === t ? "bg-brand-600 text-white" : "bg-gray-800 text-gray-400"
              }`}
            >
              {t.toUpperCase()} 로그
            </button>
          ))}
        </div>
        <JobConsole
          key={logTab}
          jobPath={`/export/logs/${logTab}`}
          onState={(s) => logTab === "fp16" ? setFp16State(s) : setOnnxState(s)}
        />
      </div>
    </div>
  );
}
