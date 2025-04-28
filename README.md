

# 🏥 AI-Based Healthcare System

An AI-powered healthcare platform that offers **voice-based diagnosis**, **imaging AI for medical scans**, and **OCR report analysis**.  
It integrates **LLama models**, **ElevenLabs**, **gTTS**, **OCR (Tesseract)**, and **medical image segmentation** to support early disease detection like **lung cancer** and **breast cancer**.

---

## ✨ Features

- 🎤 **Voice-Based Diagnosis**  
  Users provide voice inputs via ElevenLabs and gTTS. The input is processed by a LLaMA-based model to generate medical diagnostic insights.

- 🧠 **Imaging AI**  
  Image segmentation on **CT scans** and **MRI scans** for detecting abnormalities (lung cancer, breast cancer, etc.).

- 📄 **Report Reader AI**  
  OCR-powered system (using **PyTesseract**) to extract and interpret text from medical reports for automated analysis.

- 🩻 **Multimodal Output**  
  Produces diagnostic results from voice input, imaging scans, and report readings.

---

## 🛠 Tech Stack

- **Python**
- **LLaMA (Large Language Model)**
- **gTTS (Google Text-to-Speech)**
- **ElevenLabs API** (Voice Input)
- **PyTesseract** (OCR for report reading)
- **OpenCV, scikit-image** (for image segmentation)
- **TensorFlow / PyTorch** (for model training, if applicable)

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

Set up API keys (for ElevenLabs, etc.) in an `.env` file:

```env
ELEVENLABS_API_KEY=your_api_key_here
```

Install Tesseract OCR if not already installed:

```bash
# Ubuntu
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

---

## 📋 Usage

1. **Voice Diagnosis**

   - Run the voice diagnosis module:
     ```bash
     python voice_diagnosis.py
     ```
   - Speak or input your symptoms.  
   - The system will process and provide a medical assessment.

2. **Imaging AI (CT/MRI Segmentation)**

   - Run the imaging analysis:
     ```bash
     python imaging_ai.py --image path_to_scan.jpg
     ```

3. **Report Reader AI**

   - Analyze medical reports:
     ```bash
     python report_reader.py --report path_to_report.pdf
     ```

---

## 📸 Example

- Voice: "I'm feeling shortness of breath and chest pain."
- Image: CT scan showing lung nodules.
- Report: Uploaded PDF containing previous medical findings.

**Outputs:**  
- Possible diagnosis suggestions
- Segmentation map highlighting suspected regions
- Key extracted findings from OCR

---

## 🛣 Future Improvements

- Add multilingual voice support.
- Train on larger medical datasets for imaging models.
- Integration with EHR systems.
- Mobile app version.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

For queries, reach out to:  
📧 mansigambhir019@gmail.com

---


