import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import cors from "cors";

// Simple "ML" logic for the functional preview
const labels = ["Apple", "Mandarin", "Orange", "Lemon"];
const predictFruit = (mass: number, width: number, height: number, color_score: number) => {
    // Mimic the decision boundaries of the KNN model
    if (width < 6.5 && height < 6.0) return { prediction: "Mandarin", confidence: 0.95 };
    if (color_score > 0.7) return { prediction: "Lemon", confidence: 0.88 };
    if (mass > 180 && height > 9.0) return { prediction: "Orange", confidence: 0.92 };
    return { prediction: "Apple", confidence: 0.85 };
};

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(cors());
  app.use(express.json());

  // API Route: Match the Python backend's endpoint
  app.post("/predict", (req, res) => {
    const { mass, width, height, color_score } = req.body;
    
    if (mass === undefined || width === undefined || height === undefined || color_score === undefined) {
        return res.status(400).json({ error: "Missing required features" });
    }

    const result = predictFruit(mass, width, height, color_score);
    res.json(result);
  });

  // Health check
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
