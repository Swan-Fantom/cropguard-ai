/**
 * JWT auth: sign tokens and a middleware that protects routes.
 */
import jwt from "jsonwebtoken";
import { config } from "../config.js";
import { HttpError } from "./errors.js";

/** Create a signed JWT for a user id. */
export function signToken(userId) {
  return jwt.sign({ sub: userId.toString() }, config.jwtSecret, {
    expiresIn: config.jwtExpiresIn,
  });
}

/**
 * Require a valid `Authorization: Bearer <token>` header. On success attaches
 * `req.userId`; otherwise responds 401.
 */
export function requireAuth(req, _res, next) {
  const header = req.headers.authorization || "";
  const [scheme, token] = header.split(" ");
  if (scheme !== "Bearer" || !token) {
    return next(new HttpError(401, "Missing or malformed Authorization header."));
  }
  try {
    const payload = jwt.verify(token, config.jwtSecret);
    req.userId = payload.sub;
    next();
  } catch {
    next(new HttpError(401, "Invalid or expired token."));
  }
}
