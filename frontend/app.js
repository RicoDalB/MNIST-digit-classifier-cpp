const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");

const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");

const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const barsEl = document.getElementById("bars");

let drawing = false;

function initCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function getPos(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: clientX - rect.left,
    y: clientY - rect.top
  };
}

function startDraw(event) {
  drawing = true;
  const pos = getPos(event);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function draw(event) {
  if (!drawing) return;
  const pos = getPos(event);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
}

function stopDraw() {
  drawing = false;
}

function clearCanvas() {
  initCanvas();
  predictionEl.textContent = "Prediction: -";
  confidenceEl.textContent = "Confidence: -";
  barsEl.innerHTML = "";
}

function extract28x28Pixels() {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");

  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  const imageData = tempCtx.getImageData(0, 0, 28, 28).data;
  const pixels = [];

  for (let i = 0; i < imageData.length; i += 4) {
    // white drawing on black background, use red channel
    const value = imageData[i] / 255.0;
    pixels.push(value);
  }

  return pixels;
}

function renderProbabilities(probabilities) {
  barsEl.innerHTML = "";

  probabilities.forEach((p, index) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = index;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, p * 100))}%`;

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = (p * 100).toFixed(1) + "%";

    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);

    barsEl.appendChild(row);
  });
}

async function predict() {
  const pixels = extract28x28Pixels();

  const response = await fetch("http://localhost:8080/predict", {
    method: "POST",
    headers: {
      "Content-Type": "text/plain"
    },
    body: pixels.join(",")
  });

  const result = await response.json();

  if (result.error) {
    predictionEl.textContent = "Prediction: error";
    confidenceEl.textContent = result.error;
    return;
  }

  predictionEl.textContent = `Prediction: ${result.predicted_class}`;
  confidenceEl.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
  renderProbabilities(result.probabilities);
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDraw);
canvas.addEventListener("mouseleave", stopDraw);

canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", stopDraw);

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearCanvas);

initCanvas();