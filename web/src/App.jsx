/**
 * App shell: top navigation + routes. Diagnose and History are protected;
 * unauthenticated users are redirected to the login page.
 */
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { useAuth } from "./auth.jsx";
import DiagnosePage from "./pages/DiagnosePage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import RegisterPage from "./pages/RegisterPage.jsx";

function NavBar() {
  const { user, logout } = useAuth();
  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
        <Link to="/" className="flex items-center gap-2 text-lg font-semibold text-brand-700">
          <span aria-hidden>🌽</span> CropGuard AI
        </Link>
        <nav className="flex items-center gap-4 text-sm">
          {user ? (
            <>
              <Link to="/" className="text-gray-600 hover:text-brand-700">
                Diagnose
              </Link>
              <Link to="/history" className="text-gray-600 hover:text-brand-700">
                History
              </Link>
              <span className="hidden text-gray-400 sm:inline">{user.email}</span>
              <button
                onClick={logout}
                className="rounded-md border border-gray-300 px-3 py-1 text-gray-700 hover:bg-gray-50"
              >
                Log out
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="text-gray-600 hover:text-brand-700">
                Log in
              </Link>
              <Link
                to="/register"
                className="rounded-md bg-brand-600 px-3 py-1 text-white hover:bg-brand-700"
              >
                Sign up
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

/** Gate for authenticated-only routes. */
function RequireAuth({ children }) {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) {
    return <div className="mx-auto max-w-5xl px-4 py-10 text-gray-500">Loading…</div>;
  }
  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return children;
}

export default function App() {
  return (
    <div className="min-h-full">
      <NavBar />
      <main className="mx-auto max-w-5xl px-4 py-8">
        <Routes>
          <Route
            path="/"
            element={
              <RequireAuth>
                <DiagnosePage />
              </RequireAuth>
            }
          />
          <Route
            path="/history"
            element={
              <RequireAuth>
                <HistoryPage />
              </RequireAuth>
            }
          />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
