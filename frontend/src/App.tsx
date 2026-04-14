import { NavLink, Route, Routes } from "react-router-dom";
import Crawl     from "./pages/Crawl";
import Dataset   from "./pages/Dataset";
import Training  from "./pages/Training";
import Export    from "./pages/Export";
import Inference from "./pages/Inference";
import { useJobStore } from "./store/jobStore";

const navItems = [
  { to: "/",          icon: "⬇️",  label: "Crawl"    },
  { to: "/dataset",   icon: "📁",  label: "Dataset"  },
  { to: "/training",  icon: "🧠",  label: "Train"    },
  { to: "/export",    icon: "📦",  label: "Export"   },
  { to: "/inference", icon: "🔍",  label: "Inference"},
];

function RunningDot({ active }: { active: boolean }) {
  if (!active) return null;
  return <span className="ml-auto w-2 h-2 rounded-full bg-blue-400 animate-pulse" />;
}

export default function App() {
  const { crawlState, trainState, fp16State, onnxState } = useJobStore();

  return (
    <div className="flex h-screen overflow-hidden">
      {/* ── 사이드바 ─────────────────────────────── */}
      <aside className="w-48 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        <div className="px-4 py-5 border-b border-gray-800">
          <span className="text-lg font-bold text-white tracking-tight">🔭 HoloScope</span>
          <p className="text-xs text-gray-500 mt-0.5">Studio</p>
        </div>

        <nav className="flex-1 py-3 space-y-1 px-2">
          {navItems.map(({ to, icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-brand-600 text-white font-medium"
                    : "text-gray-400 hover:bg-gray-800 hover:text-white"
                }`
              }
            >
              <span>{icon}</span>
              <span>{label}</span>
              {label === "Crawl"   && <RunningDot active={crawlState === "running"} />}
              {label === "Train"   && <RunningDot active={trainState === "running"} />}
              {label === "Export"  && <RunningDot active={fp16State === "running" || onnxState === "running"} />}
            </NavLink>
          ))}
        </nav>

        <div className="px-4 py-3 border-t border-gray-800 text-xs text-gray-600">
          v1.0.0
        </div>
      </aside>

      {/* ── 메인 콘텐츠 ─────────────────────────── */}
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/"          element={<Crawl />}     />
          <Route path="/dataset"   element={<Dataset />}   />
          <Route path="/training"  element={<Training />}  />
          <Route path="/export"    element={<Export />}    />
          <Route path="/inference" element={<Inference />} />
        </Routes>
      </main>
    </div>
  );
}
