let session;

async function loadModel() {
    session = await ort.InferenceSession.create("u2net.onnx", {
        executionProviders: ["wasm"]
    });
    console.log("U²Net model loaded");
}

loadModel();

document.getElementById("upload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    const img = document.getElementById("inputImage");

    img.src = URL.createObjectURL(file);

    img.onload = async () => {
        const [tensor, width, height] = preprocessImage(img);
        const output = await session.run({ input: tensor });

        const mask = output[Object.keys(output)[0]].data;
        renderMask(mask, width, height);
    };
});

/* ------- Convert Image → ONNX Tensor ------- */
function preprocessImage(img) {
    const size = 320;

    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size).data;
    const input = new Float32Array(size * size * 3);

    for (let i = 0; i < size * size; i++) {
        input[i] = imgData[i * 4] / 255;        // R
        input[i + size * size] = imgData[i * 4 + 1] / 255;  // G
        input[i + size * size * 2] = imgData[i * 4 + 2] / 255; // B
    }

    const tensor = new ort.Tensor("float32", input, [1, 3, size, size]);
    return [tensor, img.width, img.height];
}

/* ------- Draw Mask to Canvas ------- */
function renderMask(mask, w, h) {
    const canvas = document.getElementById("maskCanvas");
    canvas.width = 320;
    canvas.height = 320;

    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(320, 320);

    for (let i = 0; i < mask.length; i++) {
        const val = mask[i] * 255;
        imageData.data[i * 4] = val;
        imageData.data[i * 4 + 1] = val;
        imageData.data[i * 4 + 2] = val;
        imageData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
}
