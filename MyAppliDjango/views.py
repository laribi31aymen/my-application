from django.shortcuts import render
from django.http import HttpResponse
import laribi_amghar_memoire
import sys
import os

def execute_python_code(request):
    output = ""
    
    # Votre code Python à exécuter
    output += "This is the output of the Python code:\n"
    output += 


import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PySimpleGUI as sg







# In[160]:


import tensorflow.lite


# In[161]:


import pickle


# In[162]:


#pour utiliser les models tensorflow lite il faut definir les entrees et sorties des models et allouer de la memoire
#Cet objet sera utilisé pour exécuter des inférences sur le modèle.
nlpinter = tensorflow.lite.Interpreter(model_path="C:/Users/lenovo/Desktop/int/CodeSource/src/models/nlp/next_word_arabic.tflite")


# In[163]:


# Cette ligne alloue de la mémoire pour les tenseurs d'entrée et de sortie du modèle TensorFlow Lite chargé dans l'objet nlpinter.
nlpinter.allocate_tensors()


# In[164]:


# Cette ligne récupère les détails sur les tenseurs d'entrée du modèle TensorFlow Lite chargé dans l'objet nlpinter.
nlp_input_details = nlpinter.get_input_details()


# In[165]:


# Cette ligne récupère les détails sur les tenseurs de sortie du modèle TensorFlow Lite chargé dans l'objet nlpinter.
nlp_output_details = nlpinter.get_output_details()

staticinter = tensorflow.lite.Interpreter(model_path="C:/Users/lenovo/Desktop/int/CodeSource/src/models/static/static_letters.tflite")
staticinter.allocate_tensors()
static_input_details = staticinter.get_input_details()
static_output_details = staticinter.get_output_details()

dynamicinter = tensorflow.lite.Interpreter(model_path="C:/Users/lenovo/Desktop/int/CodeSource/src/models/dynamic/dynamic_detection.tflite")
dynamicinter.allocate_tensors()
dynamic_input_details = dynamicinter.get_input_details()
dynamic_output_details = dynamicinter.get_output_details()


# In[166]:


#les fonctions ci dessous servent a manipuler les models facilement
def nlp_api(input):
    nlpinter.set_tensor(nlp_input_details[0]['index'], input)
    nlpinter.invoke()
    return nlpinter.get_tensor(nlp_output_details[0]['index'])


def static_api(input):
    staticinter.set_tensor(static_input_details[0]['index'], input)
    staticinter.invoke()
    return staticinter.get_tensor(static_output_details[0]['index'])


def dynamic_api(input):
    dynamicinter.set_tensor(dynamic_input_details[0]['index'], input)
    dynamicinter.invoke()
    return dynamicinter.get_tensor(dynamic_output_details[0]['index'])


# In[167]:


#le tokenizer du nlp model doit etre sauvegarde durant le training et initialise durant inference
with open('C:/Users/lenovo/Desktop/int/CodeSource/src/models/nlp/customarabictokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


# In[168]:


#nombre de mots dans le dictionnaire
total_words = len(tokenizer.word_index)


# In[169]:


#combien de mots a predire
next_words = 1


# In[170]:


#nombre maximal de mots dans une phrase defini par la longueur maximale de la phrase la plus longue durant le training
max_sequence_len = 4


# In[171]:


#padding des sequences
def pad_sequences(vec, maxlen):
    if len(vec)==0:vec=[1]
    vec = list(np.zeros(maxlen-1)) + [vec[-1]]
    return [vec]


# In[172]:


#fonction inference pour nlp
def nlpprep(seed_text):
    #convertir les mots en vecteurs
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    #padding
    token_list = pad_sequences(token_list, maxlen=max_sequence_len - 1)
    #executer inference
    predicted = nlp_api(np.array(token_list, dtype='float32'))
    #prendre seulement les mots avec une probabilite depassant un seuil
    #a experimenter apres changement du model
    sortedarr = np.flip(np.sort(predicted))[predicted > 0.01]
    nwords = sortedarr.shape[0]
    #on trie les INDEX des mots probables
    indexes = predicted.flatten().argsort()[-nwords:][::-1]
    words = []
    #on trouve le mot corespondant au nombre dans le dicionnaire du tokenizer et on le save dans une liste
    #peut etre mieux implemente en utilisant un dicitonnaire python
    for i in indexes:
        for word, index in tokenizer.word_index.items():
            if index == i:
                words.append(word)
                break
    return words


# In[173]:


#fonction de conversion des mots en nombres jai inclus un example pour la personalisation de la fonction
def tokengen(text):
    tokenlist = []
    temp = text.split(' ')
    if 'فضلك' in text:
        temp.remove('فضلك')
        temp.remove('من')
        temp.insert(0, 'من فضلك')
    tokenlist += temp
    return tokenlist


# In[174]:


#dictionnaire python contenant les lettres arabes et leurs index dans lalphabet
vocab = eval(open('C:/Users/lenovo/Desktop/int/CodeSource/src/models/static/arabic_vocab.txt', 'r', encoding='utf-8').read())
count = 1


# In[175]:


#fonction caclul angle avec coordonees x et y
def get_angle(vec):
    if vec[0] != 0:
        return np.arctan(vec[1] / vec[0])
    else:
        return 0


# In[176]:


from arabic_reshaper import reshape
import numpy as np


# In[177]:


#seuil pour detection dynamique
predictions = []
threshold = 0.3
actions = np.array(['كتاب', 'ممكن', 'طبيب', 'بخير', 'مساعدة', 'المستشفى', 'انا', 'لي', 'اسمي',
                    'الانترنت', '', 'صيدلية', 'هاتف', 'من فضلك', 'الشرطة', 'السلام', 'ادرس',
                    'اتصالات', 'الجامعة', 'اريد', 'مادا', 'انت', 'لك'])


# In[178]:


#chaque action aura un nombre, dictionnaire python
jsp = {idx: i for idx, i in enumerate(actions)}


# In[179]:


import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont


# In[180]:


#solution google pour la detection holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# In[181]:


#convertir limage en rgb et extraire les landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


# In[182]:


#dessiner les landmarks sur limage fonction optionelle
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# In[183]:


#font pour les sous titres arabes dessines sur limage
#Cette ligne définit le chemin vers le fichier de police (font) à utilise
font_path = 'C:/Users/lenovo/Desktop/int/CodeSource/src/models/static/arabic3.ttf'
font_size = 36


# In[184]:


# Cette ligne charge la police à partir du chemin spécifié (font_path) avec la taille spécifiée (font_size).
font = ImageFont.truetype(font_path, font_size)


# In[185]:


# Cette ligne importe le module hands de la bibliothèque MediaPipe, qui est utilisé pour détecter et suivre les mains dans les images ou les flux vidéo.
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# In[186]:


#fonction pour dessiner les sous titres arabes sur limage, actuellement desactivee
def toarabic(text_arabic, frame):
    text_display = reshape(text_arabic)
    text_size = font.getsize(text_display)

    text_image = Image.new('RGBA', text_size, (0, 0, 0, 0))

    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((0, 0), text_display, font=font, fill=(255, 255, 255, 255))

    text_np = np.array(text_image)

    text_cv2 = cv2.cvtColor(text_np, cv2.COLOR_RGBA2BGR)

    frame[400:400 + text_cv2.shape[0], 0:0 + text_cv2.shape[1]] = text_cv2
    return frame


# In[187]:


#fonction pour transformer les landmarks en representation geometrique vectorielle relative au points de references, voir memoire pour plus de details
def extract_keypointsstatic(result):
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        hand = [
            [landmarks[2].x - landmarks[0].x, landmarks[2].y - landmarks[0].y],
            [landmarks[3].x - landmarks[0].x, landmarks[3].y - landmarks[0].y],
            [landmarks[4].x - landmarks[0].x, landmarks[4].y - landmarks[0].y],
            [landmarks[6].x - landmarks[0].x, landmarks[6].y - landmarks[0].y],
            [landmarks[7].x - landmarks[0].x, landmarks[7].y - landmarks[0].y],
            [landmarks[8].x - landmarks[0].x, landmarks[8].y - landmarks[0].y],
            [landmarks[10].x - landmarks[0].x, landmarks[10].y - landmarks[0].y],
            [landmarks[11].x - landmarks[0].x, landmarks[11].y - landmarks[0].y],
            [landmarks[12].x - landmarks[0].x, landmarks[12].y - landmarks[0].y],
            [landmarks[14].x - landmarks[0].x, landmarks[14].y - landmarks[0].y],
            [landmarks[15].x - landmarks[0].x, landmarks[15].y - landmarks[0].y],
            [landmarks[16].x - landmarks[0].x, landmarks[16].y - landmarks[0].y],
            [landmarks[18].x - landmarks[0].x, landmarks[18].y - landmarks[0].y],
            [landmarks[19].x - landmarks[0].x, landmarks[19].y - landmarks[0].y],
            [landmarks[20].x - landmarks[0].x, landmarks[20].y - landmarks[0].y]
        ]
        return np.array(hand).flatten()
    else:
        return np.array([np.zeros(2) for i in range(15)]).flatten()


# In[188]:


#prediction model static, keypoints traites comme argument
def predictstatic(keypoints):
    prediction = static_api(np.array([[keypoints]], dtype='float32'))
    if max(prediction.flatten()) > 0.6:
        prediction = vocab[np.argmax(prediction)]
    else:
        prediction = ''
    return prediction


# In[189]:


#exactement la meme chose que la fonction pour les mains sauf que ici on prends la pose, main gauche et main droite depuis holistic
def extract_keypointsdynamic(results):
    pose, lh, rh, all = [], [], [], []

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pose = [
            [landmarks[11].x - landmarks[0].x, landmarks[11].y - landmarks[0].y],
            [landmarks[13].x - landmarks[0].x, landmarks[13].y - landmarks[0].y],
            [landmarks[15].x - landmarks[0].x, landmarks[15].y - landmarks[0].y],
            [landmarks[12].x - landmarks[0].x, landmarks[12].y - landmarks[0].y],
            [landmarks[16].x - landmarks[0].x, landmarks[16].y - landmarks[0].y],
            [landmarks[20].x - landmarks[0].x, landmarks[20].y - landmarks[0].y]
        ]
    else:
        [pose.append(np.zeros(2)) for i in range(6)]

    if results.left_hand_landmarks:
        landmarks = results.left_hand_landmarks.landmark
        lh = [
            [landmarks[2].x - landmarks[0].x, landmarks[2].y - landmarks[0].y],
            [landmarks[3].x - landmarks[0].x, landmarks[3].y - landmarks[0].y],
            [landmarks[4].x - landmarks[0].x, landmarks[4].y - landmarks[0].y],
            [landmarks[6].x - landmarks[0].x, landmarks[6].y - landmarks[0].y],
            [landmarks[7].x - landmarks[0].x, landmarks[7].y - landmarks[0].y],
            [landmarks[8].x - landmarks[0].x, landmarks[8].y - landmarks[0].y],
            [landmarks[10].x - landmarks[0].x, landmarks[10].y - landmarks[0].y],
            [landmarks[11].x - landmarks[0].x, landmarks[11].y - landmarks[0].y],
            [landmarks[12].x - landmarks[0].x, landmarks[12].y - landmarks[0].y],
            [landmarks[14].x - landmarks[0].x, landmarks[14].y - landmarks[0].y],
            [landmarks[15].x - landmarks[0].x, landmarks[15].y - landmarks[0].y],
            [landmarks[16].x - landmarks[0].x, landmarks[16].y - landmarks[0].y],
            [landmarks[18].x - landmarks[0].x, landmarks[18].y - landmarks[0].y],
            [landmarks[19].x - landmarks[0].x, landmarks[19].y - landmarks[0].y],
            [landmarks[20].x - landmarks[0].x, landmarks[20].y - landmarks[0].y]
        ]
    else:
        [lh.append(np.zeros(2)) for i in range(15)]

    if results.right_hand_landmarks:
        landmarks = results.right_hand_landmarks.landmark
        rh = [
            [landmarks[2].x - landmarks[0].x, landmarks[2].y - landmarks[0].y],
            [landmarks[3].x - landmarks[0].x, landmarks[3].y - landmarks[0].y],
            [landmarks[4].x - landmarks[0].x, landmarks[4].y - landmarks[0].y],
            [landmarks[6].x - landmarks[0].x, landmarks[6].y - landmarks[0].y],
            [landmarks[7].x - landmarks[0].x, landmarks[7].y - landmarks[0].y],
            [landmarks[8].x - landmarks[0].x, landmarks[8].y - landmarks[0].y],
            [landmarks[10].x - landmarks[0].x, landmarks[10].y - landmarks[0].y],
            [landmarks[11].x - landmarks[0].x, landmarks[11].y - landmarks[0].y],
            [landmarks[12].x - landmarks[0].x, landmarks[12].y - landmarks[0].y],
            [landmarks[14].x - landmarks[0].x, landmarks[14].y - landmarks[0].y],
            [landmarks[15].x - landmarks[0].x, landmarks[15].y - landmarks[0].y],
            [landmarks[16].x - landmarks[0].x, landmarks[16].y - landmarks[0].y],
            [landmarks[18].x - landmarks[0].x, landmarks[18].y - landmarks[0].y],
            [landmarks[19].x - landmarks[0].x, landmarks[19].y - landmarks[0].y],
            [landmarks[20].x - landmarks[0].x, landmarks[20].y - landmarks[0].y]
        ]
    else:
        [rh.append(np.zeros(2)) for i in range(15)]

    return np.array(pose + lh + rh).flatten()


# In[190]:


#extraction keypoints depuis l'image cette fonction appelle les fonctions definits au dessus
def getkeypoints(frame):
    image, results = mediapipe_detection(frame, holistic)
    draw_styled_landmarks(image, results)
    keypoints = extract_keypointsdynamic(results)
    return image, keypoints


# In[191]:


#la derniere etape pour le prediction dynamique, dynamic model et nlp combines,cette fonction appelle dautres fonctions definis au dessus
def predictdynamic(keypoints, text, words):
    res = dynamic_api(np.expand_dims(keypoints, axis=0).astype('float32'))
    #on prends le sign avec la plus grande probabilite
    predictedaction = jsp[np.argmax(res)]
    #print laction detectee et les mots possibles par le nlp
    print(predictedaction, words)
    if np.max(res) >= threshold:
        if words:
            if len(text) == 1:
                #monkey patching pour quand le model etait biase a revoir
                if predictedaction !='هاتف':
                    text.append(predictedaction)
            #si le mot detecte nest pas le dernier mot dans le phrase et le mot detecte est dans les mots possibles par le nlp on ajoute laction
            elif predictedaction != text[-1] and predictedaction in words:
                text.append(predictedaction)
        elif words is None:
            text.append(predictedaction)
        print('_______________',text)
        return text

    else:
        print('--------------',text)
        return text


# In[192]:


#ici c'est la partie du main pour lancer le programme 
#........................
#........................
#........................





#ici on utilise pysimplegui pour la creation du gui
#2 images holders, sur une seule ligne
img_row = [
    [sg.Image(key="-imgstatic-"),sg.Image(key="-imgdynamic-")],
]
#ligne de bouttons
btn_row = [
    [sg.Checkbox('Activate dynamic', key='-dactivate-'), sg.Button("dynamic"),
     sg.Checkbox('Activate nlp', key='-nlpactivate-'), sg.Button('nlp'),
    sg.Checkbox('Activate', key='-sactivate-'), sg.Button("static"),
     sg.Button("erase letter"),
     sg.Button("erase word"),
     sg.Button("erase all")],
]





#text area
text_row = [
    [sg.Multiline(size=(100, 7), key='textbox', font=('Helvetica', 24), justification='r')],
]
#on assemble le tout
layout = [
    [
        sg.Column(img_row),
    ],
    [

        sg.Column(btn_row),
    ],
    [

        sg.Column(text_row),
    ]

]
window = sg.Window("I hope this works", layout)





#object capture video opencv
vid = cv2.VideoCapture(0)





#ces variables servent a activer ou desactiver les models
count = 0
sequence = []
dynamic = False
static = False
nlpactive = False
record = False
recordstatic = True
textdynamic = [' ']
letterbuffer=''

while True:
    ret, frame = vid.read()
    event, values = window.read(timeout=10)
    #si le model dynamic est active
    if dynamic:
        #on execute la detection et on dessine sur limage
        framedynamic, resultsdynamic = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(framedynamic, resultsdynamic)
        #si une des mains est detecte on commence la detection
        if (resultsdynamic.left_hand_landmarks or resultsdynamic.right_hand_landmarks):
            record = True
        if record:
            sequence.append(extract_keypointsdynamic(resultsdynamic))
            #une fois quon a 30 images on appelle la fonction nlp si le module est actif sinon seulement le dynamic model
        if len(sequence) == 30:
            if nlpactive:
                words = nlpprep(' '.join(textdynamic))
            else:
                words = None
            textdynamic = predictdynamic(sequence, textdynamic, words)
            sequence = []
            record = False
            #mise a jour de la phrase sur le gui en temps reel
            window['textbox'].update(''.join(reversed(reshape(' '.join(textdynamic)))))
    else:
        framedynamic = frame

    #si le module static est actif on fais la meme chose mais on prends seulement la derniere image apres 30 images
    if static:

        framestatic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultstatic = hands.process(framestatic)
        if resultstatic.multi_hand_landmarks:
            [mpDraw.draw_landmarks(framestatic, x,
                                   mpHands.HAND_CONNECTIONS) for x in resultstatic.multi_hand_landmarks]
            recordstatic = True
            if recordstatic:
                count += 1
            if count == 30:
                keypointsstatic = extract_keypointsstatic(resultstatic)
                predictionstatic = predictstatic(keypointsstatic)
                letterbuffer+=predictionstatic
                textdynamic[-1]=letterbuffer
                window['textbox'].update(''.join(reversed(reshape(''.join(textdynamic)))))
                count = 0
                recordstatic = False

    else:
        framestatic = frame

    #ces boutons servent a activer ou desactiver les modules, ne pas oublier dutiliser la checkbox dabord
    if event == "dynamic" and window["-dactivate-"].get() == True:
        dynamic = True
    elif event == "dynamic" and window["-dactivate-"].get() == False:
        dynamic = False
    if event == "static" and window["-sactivate-"].get() == True:
        static = True
        textdynamic.append(letterbuffer)
    elif event == "static" and window["-sactivate-"].get() == False:
        static = False
        letterbuffer = ''
    elif event == "nlp" and window["-nlpactivate-"].get() == True:
        nlpactive = True
    elif event == "nlp" and window["-nlpactivate-"].get() == False:
        nlpactive = False
    #ces bouttons servent a supprimer tout, un seul mot ou une seule lettre, tres experimental s'attendre a des bugs
    elif event=='erase all':
        textdynamic=[' ']
        window['textbox'].update(' ')
    elif event == 'erase word':
        try:
            textdynamic = textdynamic[0:-1]
            letterbuffer=''
        except:pass
        window['textbox'].update(''.join(reversed(reshape(' '.join(textdynamic)))))
    elif event == 'erase letter':
        try:
            textdynamic[-1] = textdynamic[-1][0:-1]
            letterbuffer = letterbuffer[0:-1]
        except:pass
        window['textbox'].update(''.join(reversed(reshape(' '.join(textdynamic)))))
    #on met a jour les images des gui pour que ca soit une video
    window["-imgstatic-"].update(data=cv2.imencode('.ppm', framestatic)[1].tobytes())
    window["-imgdynamic-"].update(data=cv2.imencode('.ppm', framedynamic)[1].tobytes())

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

vid.release()
window.close()  
# Remplacez ceci par le code que vous souhaitez exécuter
    
    # Renvoyer le résultat en tant que réponse HTTP
    return HttpResponse(output)
