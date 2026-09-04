/**
 * Auth routes: register, login, and "who am I".
 *
 *   POST /api/auth/register  { email, password, name? }  -> { token, user }
 *   POST /api/auth/login     { email, password }         -> { token, user }
 *   GET  /api/auth/me        (Bearer token)              -> { user }
 */
import { Router } from "express";

import { asyncHandler, HttpError } from "../middleware/errors.js";
import { requireAuth, signToken } from "../middleware/auth.js";
import { User } from "../models/User.js";

const router = Router();

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MIN_PASSWORD_LEN = 8;

/** Validate + normalize register/login input. Throws HttpError(400) on bad input. */
function parseCredentials(body, { requireName = false } = {}) {
  const email = String(body?.email || "").trim().toLowerCase();
  const password = String(body?.password || "");
  const name = String(body?.name || "").trim();

  if (!EMAIL_RE.test(email)) {
    throw new HttpError(400, "A valid email is required.");
  }
  if (password.length < MIN_PASSWORD_LEN) {
    throw new HttpError(400, `Password must be at least ${MIN_PASSWORD_LEN} characters.`);
  }
  if (requireName && name.length > 80) {
    throw new HttpError(400, "Name is too long (max 80 characters).");
  }
  return { email, password, name };
}

router.post(
  "/register",
  asyncHandler(async (req, res) => {
    const { email, password, name } = parseCredentials(req.body, { requireName: true });

    const existing = await User.findOne({ email });
    if (existing) {
      throw new HttpError(409, "An account with that email already exists.");
    }

    const passwordHash = await User.hashPassword(password);
    const user = await User.create({ email, passwordHash, name });

    const token = signToken(user._id);
    res.status(201).json({ token, user: user.toPublicJSON() });
  })
);

router.post(
  "/login",
  asyncHandler(async (req, res) => {
    const { email, password } = parseCredentials(req.body);

    // Need the hash (select:false by default) to verify.
    const user = await User.findOne({ email }).select("+passwordHash");
    // Same generic message whether the email is unknown or the password is wrong,
    // so we don't leak which emails have accounts.
    const ok = user && (await user.verifyPassword(password));
    if (!ok) {
      throw new HttpError(401, "Invalid email or password.");
    }

    const token = signToken(user._id);
    res.json({ token, user: user.toPublicJSON() });
  })
);

router.get(
  "/me",
  requireAuth,
  asyncHandler(async (req, res) => {
    const user = await User.findById(req.userId);
    if (!user) {
      throw new HttpError(404, "User not found.");
    }
    res.json({ user: user.toPublicJSON() });
  })
);

export default router;
