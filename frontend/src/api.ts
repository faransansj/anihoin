/** API 클라이언트 — fetch 래퍼 + WebSocket 팩토리 */

const BASE = "/api";

async function req<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get:    <T>(path: string)              => req<T>("GET",    path),
  post:   <T>(path: string, body?: unknown) => req<T>("POST",   path, body),
  patch:  <T>(path: string, body?: unknown) => req<T>("PATCH",  path, body),
  delete: <T>(path: string, body?: unknown) => req<T>("DELETE", path, body),

  /** multipart/form-data 업로드 */
  upload: async <T>(path: string, form: FormData): Promise<T> => {
    const res = await fetch(`${BASE}${path}`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json() as Promise<T>;
  },

  /** WebSocket URL (ws:// or wss://) */
  ws: (path: string): WebSocket => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return new WebSocket(`${proto}://${location.host}/api${path}`);
  },
};
