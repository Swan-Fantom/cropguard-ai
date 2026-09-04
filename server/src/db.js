/**
 * MongoDB connection via Mongoose.
 */
import mongoose from "mongoose";
import { config } from "./config.js";

export async function connectDB() {
  mongoose.set("strictQuery", true);
  await mongoose.connect(config.mongoUri, {
    serverSelectionTimeoutMS: 8000, // fail fast if Mongo isn't reachable
  });
  const { host, port, name } = mongoose.connection;
  console.log(`[db] connected to MongoDB ${host}:${port}/${name}`);
  return mongoose.connection;
}
