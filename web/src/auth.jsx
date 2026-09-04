/**
 * Auth context: holds the current user, exposes login/register/logout, and
 * restores the session on load from a stored token.
 */
import { createContext, useContext, useEffect, useState } from "react";

import * as api from "./api.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true); // true while we check an existing token

  // On mount: if we have a token, fetch the user to confirm it's still valid.
  useEffect(() => {
    let active = true;
    async function restore() {
      if (!api.getToken()) {
        setLoading(false);
        return;
      }
      try {
        const { user } = await api.me();
        if (active) setUser(user);
      } catch {
        api.setToken(null); // stale/invalid token
      } finally {
        if (active) setLoading(false);
      }
    }
    restore();
    return () => {
      active = false;
    };
  }, []);

  async function login(email, password) {
    const { token, user } = await api.login(email, password);
    api.setToken(token);
    setUser(user);
  }

  async function register(email, password, name) {
    const { token, user } = await api.register(email, password, name);
    api.setToken(token);
    setUser(user);
  }

  function logout() {
    api.setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
