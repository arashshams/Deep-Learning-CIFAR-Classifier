# 🚀 CIFAR-10 Ensemble Image Classifier  

An end-to-end deep learning project for image classification using the **CIFAR-10 dataset**, featuring:

- Multiple CNN architectures  
- Transfer learning (VGG16)  
- An **ensemble prediction model**  
- A fully functional **Streamlit web app**  
- Model evaluation: confusion matrix, classification report  
- Modular and extendable project structure  

---

## 🌐 Live Demo (App)

Here is the [link](https://deep-learning-cifar-classifier-xc2cd4emw6cxuvskmpnger.streamlit.app/) to the deployed application. Feel free to give it a shot. 🎯

You can also run the app locally:  
```bash
streamlit run app/app.py
```

🖥 Streamlit App Features

Upload your own 32×32 or larger images

Generate random CIFAR-10 test samples

Ensemble prediction with confidence score

Dark mode toggle 🌙

GitHub link button

Clean 2-column UI

Refresh button for random sampling

![App Demo](reports/figures/app_demo.gif)


## 📂 Project Structure
```text
Deep-Learning-CIFAR-Classifier/
│
├── app/
│   └── app.py                   # Streamlit web application
│
├── notebooks/
│   └── CIFAR_10_Image_Classification.ipynb
│
├── src/
│   ├── predict.py               # ensemble prediction logic
│   └── data.py                  # dataset helpers (optional)
│
├── models/                      # trained models (.h5)
│
├── reports/
│   └── figures/                 # plots, app GIF, etc.
│
├── requirements.txt             
└── README.md
```




