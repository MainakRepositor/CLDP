import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as MobileNetV2_preprocess_input

# App title
st.title('Corn Leaf Diseases Detection and Remedial System')

## APP info    
st.write('''
## About

The plant diseases compose a threat to global food security and smallholder farmers whose livelihoods depend mainly on agriculture and healthy crops. 
In developing countries, smallholder farmers produce more than 80% of the agricultural production, 
and reports indicate that more than fifty percent loss in crop due to pests and diseases. 
The world population expected to grow to more than 9.7 billion by 2050, making food security a major concern in the upcoming years. Hence, rapid and accurate methods of indentying plant diseases are needed to do the appropiate measures.




''')

## load file
st.sidebar.header("File Uploads")
st.sidebar.write("In this section you can upload the files and check the outcomes")
st.sidebar.markdown(
    f'<a href="https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/data" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Reference Dataset</a>',
    unsafe_allow_html=True
)
uploaded_image = st.sidebar.file_uploader('', type=['jpg','png','jpeg'])



################### Class Dict and Dataframe of Probabilites #############################
# Map class
map_class = {
        0:'Northern Leaf Blight',
        1:'Common Rust',
        2:'Gray Leaf Spot',
        3:'Healthy'
        }
        
#Dataframe 
dict_class = {
        'Corn Leaf Condition': ['Northern Leaf Blight', 'Common Rust','Gray Leaf Spot','Healthy'],
        'Confiance': [0,0,0,0]
        }
        
df_results = pd.DataFrame(dict_class, columns = ['Corn Leaf Condition', 'Confiance'])
    
def predictions(preds):
    df_results.loc[df_results['Corn Leaf Condition'].index[0], 'Confiance'] = preds[0][0]
    df_results.loc[df_results['Corn Leaf Condition'].index[1], 'Confiance'] = preds[0][1]
    df_results.loc[df_results['Corn Leaf Condition'].index[2], 'Confiance'] = preds[0][2]
    df_results.loc[df_results['Corn Leaf Condition'].index[3], 'Confiance'] = preds[0][3]

    return (df_results)          

########################################### Load the model #########################
#@st.cache
def get_model():

    model = tf.keras.models.load_model("model_mobnetv2")
    return model

if __name__=='__main__':
    
    # Model
    model = get_model()

    # Image preprocessing
    if not uploaded_image:
        st.sidebar.write('Please upload an image before preceeding!')
        st.stop()
    else:
        # Decode the image and Predict the class
        img_as_bytes = uploaded_image.read() # Encoding image
        st.write("## Corn Leaf Image")
        st.image(img_as_bytes, use_column_width= True) # Display the image
        img = tf.io.decode_image(img_as_bytes, channels = 3) # Convert image to tensor
        img = tf.image.resize(img,(224,224)) # Resize the image
        img_arr = tf.keras.preprocessing.image.img_to_array(img) # Convert image to array
        img_arr = tf.expand_dims(img_arr, 0) # Create a bacth

    img = MobileNetV2_preprocess_input(img_arr)

    Genrate_pred = st.button("Detect Result") 
 
    if Genrate_pred:
        st.sidebar.subheader('Probabilities by Class') 
        preds = model.predict(img)
        preds_class = model.predict(img).argmax()

        st.sidebar.dataframe(predictions(preds))

        st.sidebar.info('The results are based on confidence score and hence accuracy % need not be calculated.')

        if (map_class[preds_class]=="Northern Leaf Blight") or (map_class[preds_class]=="Common Rust") or (map_class[preds_class]=="Gray Leaf Spot"): 
            st.subheader("The Corn Leaf is infected by {} disease".format(map_class[preds_class]))

        if(map_class[preds_class]=="Northern Leaf Blight"):
            st.markdown('''Northern Leaf Blight (NLB) or Corn Blight is a fungal disease affecting corn plants. Its causes include:

1. **Fungal Infection:** Blight is primarily caused by the fungus *Exserohilum turcicum* (formerly known as *Helminthosporium turcicum*).
2. **Moisture and Temperature:** Warm, humid conditions favor the development of NLB.
3. **Susceptible Corn Varieties:** Certain corn varieties are more prone to NLB infection.

Remedies and treatments for NLB often involve a combination of cultural practices, fungicides, and resistant varieties:

1. **Cultural Practices:**
   - **Crop Rotation:** Rotate crops to break the disease cycle as the fungus can survive on crop debris.
   - **Tillage:** Deep plowing can bury infected residue, reducing the fungal load.
   - **Spacing:** Proper plant spacing allows for better air circulation, minimizing moisture on leaves.
   - **Remove Infected Leaves:** Remove and destroy infected leaves to prevent further spread.

2. **Resistant Varieties:** Planting corn varieties that are genetically resistant to NLB can significantly reduce the disease impact.

3. **Fungicides:** Fungicides can be used to manage NLB, but their effectiveness depends on timing and the severity of the infection. Commonly used fungicides for NLB include:
   - **Azoxystrobin** (e.g., Quadris)
   - **Trifloxystrobin** (e.g., Flint)
   - **Propiconazole** (e.g., Tilt)

It's crucial to follow the recommended application rates and safety precautions when using fungicides.

Always consult with local agricultural extension services or professionals for specific product recommendations, application timing, and regulations in your area.''')

        elif (map_class[preds_class]=="Common Rust"):
            st.markdown('''Common Rust is a fungal disease affecting corn leaves. Its causes often involve environmental factors like high humidity, warm temperatures, and leaf wetness. Here are some suggestions for both preventing and treating Common Rust:

### Prevention:
1. **Crop Rotation:** Avoid planting corn in the same area every year. Rotate crops to reduce disease buildup in the soil.
2. **Resistant Varieties:** Choose corn varieties resistant to rust diseases.
3. **Proper Spacing:** Ensure adequate spacing between plants to promote air circulation, reducing humidity around the leaves.
4. **Fungicides:** Applying fungicides preventatively can help protect plants from rust. Fungicides like azoxystrobin, propiconazole, or tebuconazole are commonly used. Always follow the instructions on the product label.

### Treatment:
1. **Fungicides:** Apply fungicides at the first signs of infection. Repeat applications as directed on the product label.
2. **Pruning Infected Leaves:** Remove and destroy infected leaves to prevent the spread of spores.
3. **Cultural Practices:** Implement good agricultural practices, such as removing crop debris after harvest to reduce overwintering sites for the fungus.

However, when it comes to specific medicine names or products, it's essential to consult with agricultural extension services, local experts, or agronomists. They can provide recommendations based on your location, the severity of the infection, and any local regulations regarding fungicide use.

Always read and follow the instructions and safety guidelines provided by the manufacturer when using any agricultural chemicals.''')
            
        elif(map_class[preds_class]=="Gray Leaf Spot"):
            st.markdown('''Grey leaf spot is a fungal disease that commonly affects corn plants. It's caused by the fungus *Cercospora zeae-maydis*. The primary causes of this disease include:

1. **Moisture:** Extended periods of leaf wetness due to rain, irrigation, or high humidity create favorable conditions for fungal growth.
2. **Warm temperatures:** Optimal temperatures between 75°F to 85°F (24°C to 29°C) encourage the development and spread of the fungus.
3. **Residue management:** Infected crop debris left in the field can harbor the fungus, facilitating its recurrence in subsequent plantings.

Remedies for grey leaf spot typically involve a combination of cultural and chemical methods:

1. **Cultural practices:**
   - **Crop rotation:** Avoid planting corn in the same area repeatedly to reduce fungal buildup in the soil.
   - **Residue removal:** Remove and destroy infected plant debris after harvest to minimize overwintering of the fungus.
   - **Spacing:** Plant corn at recommended distances to promote air circulation, which can help reduce humidity around plants.

2. **Chemical control:**
   - **Fungicides:** Application of fungicides containing active ingredients like azoxystrobin, trifloxystrobin, or chlorothalonil can help manage the disease. Specific product names and dosages may vary based on location and regulations, so it is essential to consult with a local agricultural extension office or expert for recommendations tailored to your area.When considering fungicide use, it is crucial to follow the manufacturer\'s instructions regarding application timing, dosage, and safety precautions. Always remember to prioritize preventive measures and integrated pest management strategies to minimize the reliance on chemicals and reduce the risk of fungicide resistance. If you are dealing with grey leaf spot or any specific plant disease, contacting a local agricultural extension service or a plant pathology expert can provide tailored advice for your region and the current condition of your crop.''')

        else:
            st.subheader("The Corn Leaf is {}".format(map_class[preds_class]))