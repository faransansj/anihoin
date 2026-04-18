/**
 * Inference 페이지
 * - 이미지 드롭존
 * - Top-5 예측 바 차트
 */
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { api } from "../api";

interface Prediction {
  rank:       number;
  character:  string;
  confidence: number;
}

interface PredictResult {
  filename: string;
  top_k:    Prediction[];
}

export default function Inference() {
  const [result,   setResult]   = useState<PredictResult | null>(null);
  const [preview,  setPreview]  = useState<string | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);
      const r = await api.upload<PredictResult>("/inference/predict", form);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
  });

  const top1 = result?.top_k[0];

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-bold text-white">Inference</h1>
        <p className="text-sm text-gray-400 mt-0.5">학습된 모델로 캐릭터 분류 테스트</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* ── 왼쪽: 드롭존 및 프리뷰 ── */}
        <div className="space-y-4">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors min-h-[400px] flex flex-col items-center justify-center ${
              isDragActive
                ? "border-brand-500 bg-brand-600/10"
                : "border-gray-700 hover:border-gray-500 bg-gray-900"
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <img src={preview} alt="preview" className="max-h-full max-w-full rounded-lg object-contain" />
            ) : (
              <div className="space-y-2">
                <div className="text-4xl">🖼️</div>
                <p className="text-sm text-gray-300">
                  {isDragActive ? "이미지를 놓으세요" : "이미지를 드래그하거나 클릭하여 선택"}
                </p>
                <p className="text-xs text-gray-600">JPG · PNG · WEBP</p>
              </div>
            )}
          </div>
          
          {preview && (
            <button
              onClick={() => { setResult(null); setPreview(null); }}
              className="btn-ghost text-xs w-full"
            >
              이미지 초기화
            </button>
          )}
        </div>

        {/* ── 오른쪽: 결과 영역 ── */}
        <div className="space-y-4">
          {loading && (
            <div className="card flex items-center justify-center py-20 text-gray-400 text-sm">
              <span className="animate-pulse">분류 중...</span>
            </div>
          )}

          {error && (
            <div className="card border-red-800 bg-red-950/30 text-red-400 text-sm">
              {error}
            </div>
          )}

          {!loading && !result && !error && (
            <div className="card flex items-center justify-center py-20 text-gray-600 text-sm border-dashed">
              이미지를 업로드하면 결과가 여기에 표시됩니다.
            </div>
          )}

          {result && !loading && (
            <div className="card space-y-6">
              {/* Top-1 강조 */}
              {top1 && (
                <div className="flex items-center gap-3 p-4 bg-brand-600/10 rounded-lg border border-brand-600/30">
                  <span className="text-3xl font-bold text-brand-400">
                    {(top1.confidence * 100).toFixed(1)}%
                  </span>
                  <div>
                    <p className="font-semibold text-lg text-white">{top1.character.replaceAll("_", " ")}</p>
                    <p className="text-xs text-gray-500">Best match</p>
                  </div>
                </div>
              )}

              {/* Top-5 바 */}
              <div className="space-y-3">
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">Top-5 Predictions</p>
                {result.top_k.map((p) => (
                  <div key={p.rank} className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-400">
                      <span className={p.rank === 1 ? "text-gray-200 font-medium" : ""}>
                        {p.rank}. {p.character.replaceAll("_", " ")}
                      </span>
                      <span>{(p.confidence * 100).toFixed(2)}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          p.rank === 1 ? "bg-brand-500" : "bg-gray-600"
                        }`}
                        style={{ width: `${p.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
