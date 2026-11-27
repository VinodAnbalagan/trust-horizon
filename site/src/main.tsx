import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
// GitHub Pages only: the Archivo font. The course version uses the system font.
import "@fontsource-variable/archivo/wdth.css";
import "@fontsource-variable/archivo/wdth-italic.css";
import "./fonts-github.css";

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
