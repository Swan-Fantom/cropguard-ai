/**
 * User model. Passwords are stored ONLY as a bcrypt hash, never in plaintext.
 */
import bcrypt from "bcryptjs";
import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
      index: true,
    },
    // The bcrypt hash. `select: false` keeps it out of query results by default,
    // so it can't be leaked accidentally by a route that returns a user.
    passwordHash: {
      type: String,
      required: true,
      select: false,
    },
    name: {
      type: String,
      trim: true,
      default: "",
    },
  },
  { timestamps: true }
);

/** Hash a plaintext password (used at registration). */
userSchema.statics.hashPassword = function hashPassword(plain) {
  const SALT_ROUNDS = 12;
  return bcrypt.hash(plain, SALT_ROUNDS);
};

/** Compare a plaintext attempt against this user's stored hash. */
userSchema.methods.verifyPassword = function verifyPassword(plain) {
  return bcrypt.compare(plain, this.passwordHash);
};

/** Safe, public-facing shape (never includes the hash). */
userSchema.methods.toPublicJSON = function toPublicJSON() {
  return {
    id: this._id.toString(),
    email: this.email,
    name: this.name,
    createdAt: this.createdAt,
  };
};

export const User = mongoose.model("User", userSchema);
