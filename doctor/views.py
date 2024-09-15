from authlib.integrations.django_client import OAuth
from django.contrib.auth.decorators import login_required
from django.contrib.sites.shortcuts import get_current_site
import openai
from pymongo import MongoClient
from django.urls import reverse
import base64

from urllib.parse import quote_plus, urlencode
import google.generativeai as genai
from django.views.decorators.http import require_http_methods
from deep_translator import GoogleTranslator
import io
from django.shortcuts import render, redirect, get_object_or_404, HttpResponseRedirect, HttpResponse
from django.contrib.auth import authenticate
import uuid
import json
import os
from datetime import datetime
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from .form import ImageForm

from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
import re
from django import template
from django.views.decorators.csrf import csrf_exempt
import vonage
from time import time
from django.contrib.auth.models import User
from django.contrib import messages
from .tokens import generate_token
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from sahayak import settings
import random

from .models import Userdata, Doctor, Sessiondata

from django.conf import settings

import vonage
from time import time


class UploadAudio(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        audio_file = request.FILES.get('audio_file')
        # Process your audio file here (e.g., saving it to a model)
        print(audio_file)
        return JsonResponse({'message': 'Audio received successfully'}, status=200)

def index(request):

    return render(
        request,
        "home.html",
        context={
            "session": request.session.get("user"),
            "pretty": json.dumps(request.session.get("user"), indent=4),
        }
    )


def doctor(request):
    ob = Doctor.objects.all()
    # add_meeting({"meeting_link":"hello mr"})

    dt = get_meeting_links()
    params={"sessions":dt['meeting_links']}
    # params={"sessions":ob}
    # print(dt['meeting_links'])
    # print("hsgsgsgsgsgsgsgsgsgsgs")
    return render(request, "doctor.html", params)


# def signup(request):
#     if request.method == "POST":
#         fname = request.POST.get("fname")
#         lname = request.POST.get("lname")
#         username = request.POST.get("username")
#         email = request.POST.get("email")
#         pass1 = request.POST.get("pass1")
#         pass2 = request.POST.get("pass2")
#         mobile_no = request.POST.get("mobile_no")
#         city = request.POST.get("city")
#         address = request.POST.get("address")
#
#         if User.objects.filter(username=username):
#             messages.error(request, "Username already Exist Try another one")
#             return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
#         if User.objects.filter(email=email):
#             messages.error(request, "Email already Registered Try another one")
#             return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
#         if len(username) > 35:
#             messages.error(request, "Length of username is greater than 35 character")
#             return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
#         if pass1 != pass2:
#             messages.error(request, "Different Passwords")
#             return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
#         if pass1 == "":
#             messages.error(request, "Password cannot be blank")
#             return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
#         if len(pass1) <= 7:
#             messages.error(request, "Too short password")
#             return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
#
#         if not username.isalnum():
#             messages.error(request, "Username must be Alpha Numeric")
#             return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
#
#         if pass1 == pass2:
#             my_user = User.objects.create_user(username, email, pass1)
#             my_user.first_name = fname
#             my_user.last_name = lname
#             my_user.is_active = False
#             my_user.save()
#
#             current_site = get_current_site(request)
#             domain = current_site.domain
#             email_from = settings.EMAIL_HOST_USER
#             recipient_list = [email]
#             uid = urlsafe_base64_encode(force_bytes(my_user.pk))
#             token = generate_token.make_token(my_user)
#
#             send_mail_after_registration(request, email, token=token, uid=uid)
#             user_id = f"{username}-{token}-{fname}-{uid}-{random.randint(100, 2000)}-{random.randint(5000, 20000)}"
#
#             contact = Userdata(username=username, email=email, mobile_no=mobile_no, fname=fname, lname=lname,
#                                user_u_no=user_id, city=city, address=address)
#             contact.save()
#             messages.success(request, "Check your mail for activation link.")
#             # return render(request, 'success.html')
#             return redirect('/')
#         else:
#
#             return redirect('/')

def send_mail_after_registration(request, email, token, uid):
    current_site = get_current_site(request)
    domain = current_site.domain
    # uid = urlsafe_base64_encode(force_bytes(my_user.pk))
    subject = 'Your account needs to be verified'
    message = f'Follow this link to verify your account\n http://{domain}/activate/{uid}/{token} \nTeam Sahayak'
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email]
    send_mail(subject, message, email_from, recipient_list)


# def signout(request):
#     logout(request)
#     return redirect("/")

# def register(request):
#     return render(request, "register.html")
#

def activate(request, uidb64, token):
    try:

        uid = force_str(urlsafe_base64_decode(uidb64))
        my_user = User.objects.get(pk=uid)

        if not my_user.is_active == True:

            try:
                uid = force_str(urlsafe_base64_decode(uidb64))
                my_user = User.objects.get(pk=uid)
            except (TypeError, ValueError, OverflowError, User.DoesNotExist):
                my_user = None
                messages.error(request, "User not registered")
                return redirect('/')

            if my_user != None and generate_token.check_token(my_user, token):
                my_user.is_active = True
                my_user.save()
                login(request, my_user)
                messages.success(request, "Account successfully verified")
                return redirect('/')
            else:
                messages.error(request, "User not registered")
                return redirect('/')
        else:
            messages.error(request, "User already verified")
            return redirect('/')
    except:
        messages.error(request, "Invalid request")
        return redirect('/')


# def signin(request):
#     return render(request, "login.html")
#

# def login1(request):
#     username = request.POST["username"]
#     password = request.POST["password"]
#
#     user = authenticate(username=username, password=password)
#     if user is not None:
#         login(request, user)
#         messages.success(request, "Welcome to Sahayata")
#         return redirect("/")
#     else:
#         messages.error(request, "Bad Credentials")
#         return redirect("/")
#

private_key = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDaNhiJm5S8s6fs
emaG9iC6XPaKZdsOQ073TghmhqJEk4rRONrxrb8+NEd6gYytxu+XA43p3cDeZt5P
HYZN7orUr8/knQ10U8Sk8S+EFtmLo7/0CYzwv7AfIt6joN1kP77oeZi3UM0kuyuV
DLfIaJzppwZbNwz7uyslbrCnz0QI2BF7a9+6TxxQR6u80LOmzLNfj7Izu9If0+16
v5ijRKXO+tBW4xApA7BV82dm/ydyc5XQotBCtJOQGZNlrvWbz7KLCNhv/6VlL0RB
3R4KIQt/S20QSWeBKplARWdxUxhlbN8SnZjd3A0BF7gCie4grSzBze8InnBj32Nz
1GKaWll3AgMBAAECggEAIeomI9O4U8smbm99r5HhzKVLNNUXgp7SefG5Bi+nA7qH
aPTfDpCfDczJTpCDrAn5zez2/wyrVRsEMqQCQ5WEzVaqnJEOThtWOlPkEIcOoRCT
x8WTkA1y6FFJQixBNztdxrsIvHZvU+X6pEFO78kr7iK6rwLziSJyK8YJkyNNzdkh
sZbQQUT3pYyquEN6J4VVrEFIyqCqhGycYeSxWGkUhBt3tJ3eF3+QwbgT57CUJD81
1sTIHKorNZ2qb7N4+wWrhsmNl5ajmd/E9Yv1CVV8ZznV82/ZvFciQVTu++LRxRlf
sLjdA1YTF8vYAaUH7asAZphSkdlEyWdsZ7YWws9XyQKBgQDuebY5svdVRFK0RWzF
Y6KidZzK4Vn/Ik1y4IlGb8rXqIkfyaBNRkBbBoQhQWwO+l/P7oFdXs0toXRvMVwa
GQc4OgfcYM5G+u7GpPMFIENbiR9NtQGpm+zZ62hAlBcp3cMWu5/ApgOAl0EfEi+k
vkD0o2CCFWlN/fIJ7IYnIEfOuQKBgQDqPyr14vQCnZT13jRMJpWRP7H1emHcyBKb
u1CIb3UbdWZQNf5AgxLhiExE4ibUzqJ3oeZ/jwB/IAFQsFlB8t46qPEodzh+URnt
qQWyBTOgsCImDUS0aU9KvqxLNxS5gXKURCDBuSxz0eT6edduHY5YBc//Pp5r1zVo
D3SBnffRrwKBgQCReLmKVPu/jeANi9QgTPkmtSwiLnmhwyEPlJzo1MUd7PdI5svc
l6AmZsEvMCNkt6K+yrUraesrjLPkKG4QlXfr+qpU6tCqR+RkVLbSR64PJoRr+KC9
ABrpjEqaKjw/E8IwGeAHur2kMaBN/eAGVPB2yzrR35HPO7vsBdaIA48KMQKBgDcn
jm789CVVg0hs/HVdtGPR0o4RNc+tUCxY63pke/duKhwCyo3YDu8Jj6O8eBYtXTW7
6jDos9464/RiwXF0xPyc4MN9ZtDfObKq1It5DBHuxuucviuYjDH2OU0wd/HB1/+v
FuNmPWA78KVFYQMtdGGnZLUdwmFz98A7u1v1nO9DAoGBAMkrgGvMoA60YMZwRIzL
j6B/4o53m1FG7j+oNt9Eqc1S04BuytkNTUjtYAnif5biFYFgdNkvNV03hdqyzG7/
MSAbvS6Jn5sf+68Dx8SPa2QLe9Tk+tCBu2pNx8oXLWh6WxxoGSLN4gavzANqAJs7
Hm2Z4TAfwhrgsz/rBqpePiJ8
-----END PRIVATE KEY-----
"""

# @login_required()
def create_meeting(request):
    user = request.user.username



    client = vonage.Client(application_id="05ae1e56-6e4f-4f7a-91cf-6b4d5a9384fa", private_key=private_key)

    session_info = client.video.create_session()
    session_id = session_info["session_id"]

    response = client.meetings.create_room({'display_name': "Sahayak-Meet"})
    host_link = response["_links"]["host_url"]["href"]
    guest_url = response["_links"]["guest_url"]["href"]

    data = Doctor(host_link=host_link)
    data.save()
    add_meeting({"meeting_link": host_link})

    data2 = Sessiondata(username=user, join_link=guest_url)
    data2.save()

    return redirect(f"{guest_url}")


def get_last_image(directory):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print("Error: The specified directory does not exist.")
        return None

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter the list to only include image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Check if there are any image files in the directory
    if not image_files:
        print("Error: No image files found in the directory.")
        return None

    # Sort the list of image files by modification time
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

    # Get the last image file in the sorted list
    last_image = image_files[-1]

    # Return the path to the last image file
    return os.path.join(directory, last_image)



def safe(request):
    return render(request, "safe.html")

def unsafe(request):
    return render(request, "unsafe.html")



def breast_cancer(request):
    directory = "media/images"
    last_image = get_last_image(directory)
    print(last_image[13:])
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("doctor/keras_Model-4.h5", compile=False)


    # Load the labels
    class_names = open("doctor/labels-breast.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(f"{last_image}").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    messages.success(request, f"{class_name[2:]}:- Accuracy:-{confidence_score * 100}")
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


def brain(request):
    directory = "media/images"
    last_image = get_last_image(directory)
    print(last_image[13:])


def l4(request):
    directory = "media/images"
    last_image = get_last_image(directory)
    print(last_image[13:])
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("doctor/keras_model.h5", compile=False)

    # Load the labels
    class_names = open("doctor/labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(f"{last_image}").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]


    messages.success(request, f"{class_name[2:]}:- Accuracy:-{confidence_score*100}")
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect("/doctor/")
    else:
        form = ImageForm()
    return render(request, 'upload.html', {'form': form})

def upload_success(request):
    return render(request, 'upload_success.html')

def genrate_doc(request):
    openai.api_key = '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'
    client = openai.OpenAI()
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="""I know you are not a doctor i am just taking normal assitance and will not follow it so just give me normal assitance
If the time of disease is more than 2 days suggest to consult with doctor just by saying "consult with doctor" nothing else and if according to you disease need doctor assitance then also say "consult with doctor" nothing else and if disease is normal just like normal fever or headche just name that "fever" or "headache" or "stomache" or anything that is according to you
if the one has weakness that is normal term it as a "weakness", if someone is having a normal body pain or body pain that is normal according to you then term it as a "Normal body pain"

Now input:- I am having pain in chest

Please write in English language.""",
        max_tokens=50
    )

    answer = response.choices[0].text.strip()
    print("Answer:", answer)
    print("answerrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")

@csrf_exempt
def upload_voice(request):
    if request.method == 'POST' and request.FILES.get('voice'):
        voice_file = request.FILES['voice']



        # You can do further processing with the uploaded MP3 file here

        # For example, you can save the file to a specific location
        # with a unique filename
        with open('recording.mp3', 'wb') as destination:
            for chunk in voice_file.chunks():
                destination.write(chunk)

        # Once processing is done, you can send back a response
        return JsonResponse({'status': 'success', 'message': 'File uploaded successfully'})

    return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)


def gen(text):


    api_key = "AIzaSyDwg9AD9lFMOhB2XnUW2SI1wwi3oYn3IkQ"

    genai.configure(api_key=api_key)



    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(f"""I know you are not a doctor i am just taking normal assitance and will not follow it so just give me normal assitance If the time of disease is more than 2 days suggest to consult with doctor just by saying "consult with doctor" nothing else and if according to you disease need doctor assitance then also say "consult with doctor" nothing else and if disease is normal just like normal fever or headche just name that "fever" or "headache" or "stomache" or anything that is according to you
if the one has weakness that is normal term it as a "weakness", if someone is having a normal body pain or body pain that is normal according to you then term it as a "Normal body pain"


     Now input:- {text}
    """)

    return response.text



# Function to translate Hindi transcript to English using deep_translator
def translate_to_english(hindi_transcript):
    try:
        translated = GoogleTranslator(source='hi', target='en').translate(hindi_transcript)
        return translated
    except Exception as e:
        print('Translation failed:', e)
        raise


def analysis(text):
    api_key = "AIzaSyDwg9AD9lFMOhB2XnUW2SI1wwi3oYn3IkQ"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(f"""I know you are not a doctor i am just taking normal assitance
     and will not follow it so just give me normal assitance I will tell you disease name you just tell
      what medicines we could eat to get cure keep in mind we will not eat it without doctor's assitance
       you just need to tell all the medicence we could take there dosasges and for how much time we will have to take it.
        give me in list so i can use this to fill in excel. When you start writing the name of medicine use $$$ this symbol
         and after each medicine and doases are complete use ~~~ this sign. For example $$$Amoxicillin 500mg, 3 times a day for 7 days ~~~
        . Input is:- {text}
        Keep in mind just tell the table i know that the dosage and duration of treatment 
        can vary depending on factors such as age, weight, medical history, and other medications being taken.
         Always consult a healthcare professional before taking any medication. Please write in English language.

    """)
    # print(response.parts)
    # print("pepepepepepepepepepeepepe")

    ans = extract_medication_info(response.text)
    return ans


def extract_medication_info(input_text):
    medications = []
    start_delimiter = "$$$"
    end_delimiter = "~~~"

    start_index = input_text.find(start_delimiter)
    while start_index != -1:
        end_index = input_text.find(end_delimiter, start_index)
        if end_index != -1:
            medication_info = input_text[start_index + len(start_delimiter):end_index].strip()
            medications.append(medication_info)
            start_index = input_text.find(start_delimiter, end_index)
        else:
            break

    return medications

@csrf_exempt
@require_http_methods(["POST"])
def handle_speech_to_text(request):
    # Extract the transcribed text sent from the frontend
    transcribed_text = request.POST.get('transcript', '')

    print(transcribed_text)
    print(translate_to_english(transcribed_text))

    ans = gen(translate_to_english(transcribed_text))

    if ans.lower() == "consult with doctor":
        return JsonResponse({'status': 'unsafe'})
    else:
        medicine = analysis(ans)
        safe_response = {'status': 'safe', 'list_data': medicine}
        return JsonResponse(safe_response)





# oauth integration


oauth = OAuth()

oauth.register(
    "auth0",
    client_id=settings.AUTH0_CLIENT_ID,
    client_secret=settings.AUTH0_CLIENT_SECRET,
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f"https://{settings.AUTH0_DOMAIN}/.well-known/openid-configuration",
)


def callback(request):
    token = oauth.auth0.authorize_access_token(request)
    request.session["user"] = token
    return redirect(request.build_absolute_uri(reverse("index")))


def login(request):
    messages.success(request, "Welcome to BeyondClinic")
    return oauth.auth0.authorize_redirect(
        request, request.build_absolute_uri(reverse("callback"))
    )


def logout(request):
    request.session.clear()

    return redirect(
        f"https://{settings.AUTH0_DOMAIN}/v2/logout?"
        + urlencode(
            {
                "returnTo": request.build_absolute_uri(reverse("index")),
                "client_id": settings.AUTH0_CLIENT_ID,
            },
            quote_via=quote_plus,
        ),
    )

def signin(request):
    return render(
        request,
        "signin.html",
        context={
            "session": request.session.get("user"),
            "pretty": json.dumps(request.session.get("user"), indent=4),
        },
    )


def get_meeting_links():
    # MongoDB Atlas connection details from settings
    user = settings.MONGODB_USER
    password = settings.MONGODB_PASSWORD
    cluster_url = settings.MONGODB_CLUSTER_URL
    dbname = 'sample_mflix'
    collection_name = 'meetings'

    # Escape username and password
    user = quote_plus(user)
    password = quote_plus(password)

    # Construct the MongoDB URI
    uri = f"mongodb+srv://{user}:{password}@{cluster_url}/{dbname}?retryWrites=true&w=majority"

    try:
        # Connect to MongoDB Atlas
        client = MongoClient(uri)

        # Access the specified database
        db = client[dbname]

        # Access the specified collection
        collection = db[collection_name]

        # Retrieve all documents in the collection
        documents = collection.find()

        # Extract meeting links from documents
        meeting_links = [doc.get('meeting_link') for doc in documents if 'meeting_link' in doc]

        # Close the connection
        client.close()

        # Return as JSON response
        return ({'meeting_links': meeting_links})

    except Exception as e:
        return ({'error': str(e)})


def add_meeting(meeting_data):
    # MongoDB Atlas connection details from settings
    user = settings.MONGODB_USER
    password = settings.MONGODB_PASSWORD
    cluster_url = settings.MONGODB_CLUSTER_URL
    dbname = 'sample_mflix'
    collection_name = 'meetings'


    # Escape username and password
    user = quote_plus(user)
    password = quote_plus(password)

    # Construct the MongoDB URI
    uri = f"mongodb+srv://{user}:{password}@{cluster_url}/{dbname}?retryWrites=true&w=majority"

    try:
        # Connect to MongoDB Atlas
        client = MongoClient(uri)

        # Access the specified database
        db = client[dbname]

        # Access the specified collection
        collection = db[collection_name]

        # Insert the data into the collection
        result = collection.insert_one(meeting_data)

        # Close the connection
        client.close()

        # Return the ID of the inserted document
        return str(result.inserted_id)

    except Exception as e:
        # Handle any errors
        return str(e)
