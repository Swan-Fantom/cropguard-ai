/**
 * Central configuration, loaded once from environment variables.
 * Import `config` anywhere instead of reading process.env scattered around.
 */
import dotenv from "dotenv";

dotenv.config();

function required(name, fallback) {
  const v = process.env[name] ?? fallback;
  if (v === undefined || v === "") {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return v;
}

export const config = {
  port: parseInt(process.env.PORT || "4000", 10),
  mongoUri: process.env.MONGO_URI || "mongodb://127.0.0.1:27017/cropguard",

  // JWT_SECRET has an insecure dev default on purpose so `npm run dev` works out
  // of the box, but we warn loudly (see index.js) if it's left unchanged.
  jwtSecret: process.env.JWT_SECRET || "dev-insecure-secret-change-me",
  jwtExpiresIn: process.env.JWT_EXPIRES_IN || "7d",

  mlServiceUrl: (process.env.ML_SERVICE_URL || "http://127.0.0.1:8000").replace(/\/$/, ""),

  corsOrigins: (process.env.CORS_ORIGINS || "http://localhost:5173,http://127.0.0.1:5173")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),

  maxUploadMb: parseFloat(process.env.MAX_UPLOAD_MB || "10"),
};

export const isInsecureJwtSecret =
  config.jwtSecret === "dev-insecure-secret-change-me";
