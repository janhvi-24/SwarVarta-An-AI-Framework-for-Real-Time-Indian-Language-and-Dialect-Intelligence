# app.py
import streamlit as st
from scripts.inference import predict

st.set_page_config(page_title='SwarVarta — Hinglish Demo')
st.title('SwarVarta — Hinglish Intent & Slot Demo')

input_type = st.radio('Input type', ['Text', 'Upload audio'])
utterance = ''
if input_type == 'Text':
    utterance = st.text_area('Utterance', value='mujhe ek flight book karni hai from delhi to mumbai tomorrow')
else:
    file = st.file_uploader('Upload audio (wav/mp3)', type=['wav','mp3','m4a'])
    if file is not None:
        st.write('Transcribing...')
        try:
            import whisper, tempfile
            model = whisper.load_model('base')
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tf.write(file.read()); tf.flush()
            res = model.transcribe(tf.name)
            utterance = res['text']
            st.write('Transcription (whisper):', utterance)
        except Exception:
            st.warning('Whisper not available. Install openai-whisper for audio transcription.')

if st.button('Predict'):
    if not utterance:
        st.error('Provide text or audio first.')
    else:
        with st.spinner('Running inference...'):
            out = predict(utterance)
        st.subheader('Intent')
        st.write(out.get('intent'))
        st.subheader('Slots')
        slots = out.get('slots', [])
        if slots:
            for s in slots:
                st.write(f"{s['entity']}: {s['value']} (chars {s['start']}-{s['end']})")
        else:
            st.write('No slots found')
