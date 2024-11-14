from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import SignatureUploadForm, RegistrationForm, LoginForm
from keras.models import load_model
import numpy as np
from django.core.files.base import ContentFile
import io

# Load your pre-trained model
model = load_model('C:\\Users\\ranji\\Desktop\\infosys\\infosys\\bi_rnn_signature_verification_model_moretrained.h5')
import cv2
import numpy as np



def preprocess_image(image, img_size=(128, 128), patch_size=(128, 128)):
    # Read the image and convert it to grayscale
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image.")
    
    # Resize the image
    img = cv2.resize(img, img_size)
    
    # Extract patches
    patches = img_to_patches(img, patch_size)
    return patches



def img_to_patches(img, patch_size=(128, 128)):
    patches = []
    for i in range(0, img.shape[0], patch_size[0]):
        for j in range(0, img.shape[1], patch_size[1]):
            patch = img[i:i + patch_size[0], j:j + patch_size[1]].flatten()
            patches.append(patch)
    return np.array(patches)



def predict_signature(model, image):
    # Preprocess the uploaded image to extract patches
    patches = preprocess_image(image)  # Extract patches from the uploaded image
    patches = patches.reshape(-1, patches.shape[1])  # Flatten patches to shape (num_patches, features)

    # Reshape the input to match the model's expected input shape
    timesteps = patches.shape[0]  # Number of patches
    features = patches.shape[1]    # Number of features (flattened size)
    
    # Ensure the input shape is (1, timesteps, features)
    input_data = patches.reshape(1, timesteps, features)  # Add batch dimension

    # Make prediction
    prediction = model.predict(input_data)
    
    return np.argmax(prediction), np.max(prediction) 



def login_view(request):
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('upload_signature')  # Redirect to upload page
            else:
                form.add_error(None, 'Invalid username or password.')
    return render(request, 'login.html', {'form': form})



def registration_view(request):
    form = RegistrationForm()
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            return redirect('login')  
    return render(request, 'registration.html', {'form': form})



@login_required
def upload_signature(request):
    result = None
    confidence = None
    form = SignatureUploadForm()
    if request.method == 'POST':
        form = SignatureUploadForm(request.POST, request.FILES)
        if form.is_valid():
            signature = request.FILES['signature']
            prediction, confidence = predict_signature(model, signature)
            class_names = ['Real', 'Forged']
            result = class_names[prediction]
    return render(request, 'upload.html', {'form': form, 'result': result, 'confidence': confidence})

def logout_view(request):
    logout(request)
    return render(request, 'logout.html')