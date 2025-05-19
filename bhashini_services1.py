import base64
import requests
import json
from datetime import datetime
import os
import wave
import shutil
import time
 
 
class Bhashini_master:
   
    def __init__(self,
                 url,
                 authorization_key,
                 ulca_api_key,
                 ulca_userid
                 ):
        # print("#################### Using bhashini key ####################",ulca_api_key)
        self.translated_content = None
        self.url = url
        self.authorization_key = authorization_key
        self.ulca_api_key = ulca_api_key
        self.ulca_userid = ulca_userid
        self.asr_service_ids = {
                                    "en" : "ai4bharat/whisper-medium-en--gpu--t4",
                                    "bn" : "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
                                    "hi" : "ai4bharat/conformer-hi-gpu--t4",
                                    "te" : "ai4bharat/conformer-multilingual-dravidian-gpu--t4",
                                    "pa" : "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
                                    "ks" : "ai4bharat/conformer-multilingual-dravidian-gpu--t4",
                                }
        service1 = "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4"
        self.tts_service_id = {
                               "en": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
                               "hi":service1,
                               "bn":service1,
                               "pa":service1,
                               "ks":service1,
                               "te" : "titleai4bharat/indic-tts-coqui-dravidian-gpu--t4"
                               }
        # for converting audio_bytes to wav file
        self.sample_rate = 44100
        self.num_channels = 1  
        self.sample_width = 2
                   
    def transcribe_audio(self, audio_content, source_language):
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {"sourceLanguage": source_language},
                        "serviceId": self.asr_service_ids.get(source_language, "default-service-id"),
                        "audioFormat": "flac",
                        "samplingRate": 16000
                    }
                }
            ],
            "inputData": {
                "audio": [{"audioContent": audio_base64}]
            },
            "output": ""
        }

        headers = {
            'Content-Type': 'application/json',
            'userID': self.ulca_userid,
            'ulcaApiKey': self.ulca_api_key,
            'Authorization': self.authorization_key,
        }

        response = requests.post(self.url, headers=headers, json=payload)
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")

        if response.status_code != 200:
            # Log error and return a descriptive error message
            return f"Error: transcription request failed with status code {response.status_code}"

        try:
            data = response.json()
            transcription = data["pipelineResponse"][0]["output"][0]["source"]
        except (KeyError, json.JSONDecodeError) as e:
            print("Error parsing response:", e)
            transcription = "Error: Could not parse transcription."
        
        return transcription
 
    def translate_text(self,input_text,source_language,target_language):
       
        translation_service_id = "ai4bharat/indictrans-v2-all-gpu--t4"
       
        headers =  {        
            "Authorization" : self.authorization_key,
            "Content-Type": "application/json",
        }
 
        payload = {
                    "pipelineTasks": [
                        {
                            "taskType": "translation",
                                    "config": {
                                        "language": {
                                            "sourceLanguage": source_language,
                                            "targetLanguage": target_language
                                        },
                                        "serviceId": translation_service_id
                                    }
                                }
                            ],
                            "inputData": {
                                "input": [
                                    {
                                        "source": input_text
                                    }
                                ]
                            }
                        }
 
        payload_json= json.dumps(payload)
        try:
            response = requests.post(self.url, headers=headers, data=payload_json)
            if response.status_code ==200:
                response=response.json()
                return response["pipelineResponse"][0]["output"][0]["target"]
            elif response.status_code==504:
                # incase of timeout wait for 5 sec then retry
                time.sleep(5)
                response = requests.post(self.url, headers=headers, data=payload_json)
                response = response.json()
                return response["pipelineResponse"][0]["output"][0]["target"] if response.status_code==200 else "unresponsive bhashini"
            else:
                print("Request failed with status code: {} full response = {} ".format(response.status_code,response))
                return None
        except Exception as e:
            print("Error in translate_text function - ",e)
            return " unresponsive bhashini "
 
    def save_audio_as_wav(self,audio_bytes, directory, file_name):
 
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
 
        # Full file path
        file_path = os.path.join(directory, file_name)
 
        # Write the WAV file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.num_channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)
 
        print(f"WAV file created successfully at {file_path}.")
        return file_path
   
    def detect_language(self, audio_bytes):
        if not audio_bytes:
            return None
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        payload = {
            "pipelineTasks": [
                {"taskType": "audio-lang-detection", "config": {"serviceId": "ai4bharat/lang-detect"}}
            ],
            "inputData": {"audio": [{"audioContent": audio_base64}]}
        }
        headers = {
            'Content-Type': 'application/json',
            'userID': self.ulca_userid,
            'ulcaApiKey': self.ulca_api_key,
            'Authorization': self.authorization_key,
        }
        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            try:
                lang_code = response.json()["pipelineResponse"][0]["output"][0]["langPrediction"][0]["langCode"]
                return lang_code
            except (KeyError, IndexError):
                return "Error parsing response"
        else:
            return f"Request failed with status code: {response.status_code}"
    def detect_text_language(self, input_text):
        """
        Detects the language of the given text using Bhashini's text language detection API.
        """
        headers = {
            "Authorization": self.authorization_key,
            "Content-Type": "application/json",
        }

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "txt-lang-detection",
                    "config": {
                        "serviceId": "ai4bharat/ulca-text-lang-detection"
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": input_text
                    }
                ]
            }
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code == 200:
            try:
                response_data = response.json()
                lang_info = response_data["pipelineResponse"][0]["output"][0]["langPrediction"][0]
                lang_code = lang_info["langCode"]
                script_code = lang_info.get("scriptCode", "Unknown")
                lang_score = lang_info.get("langScore", "N/A")

                return {
                    "language_code": lang_code,
                    "script_code": script_code,
                    "confidence_score": lang_score
                }
            except (KeyError, IndexError):
                return {"error": "Error parsing response"}
        else:
            return {"error": f"Request failed with status code: {response.status_code}"}

    def base64_to_wav(self,base64_data,temp_save_path):
        binary_data = base64.b64decode(base64_data)
        date_string = datetime.now().strftime("%d%m%Y%H%M%S")
        os.makedirs(temp_save_path,exist_ok=True)
        file_path = os.path.join(temp_save_path,date_string+"_audio.wav")
        with open(file_path, "wb") as wav_file:
            wav_file.write(binary_data)
        return file_path
 
    def speak(self,content,source_language):
       
        temp_save_path = "tts_logs"
        headers = {
            "Authorization" : self.authorization_key,
            "Content-Type": "application/json",
        }
 
        payload= {"pipelineTasks": [      
                        {
                            "taskType": "tts",
                            "config": {
                                        "language": {
                                                    "sourceLanguage": source_language
                                                    },
                                        
                                        "serviceId": self.tts_service_id[source_language],
                                        "gender": "female"
                                        }
                        }
                    ],
                    "inputData": {
                        "input": [
                            {
                                "source": content
                            }
                        ]
                    }
                    }
 
        payload_json= json.dumps(payload)
 
        response = requests.post(self.url, headers=headers, data=payload_json)
        print(f"raw response = {response}")
        if response.status_code != 200:
            # Log error details and return an error message
            error_message = f"Error: TTS request failed with status code {response.status_code}. Response: {response.text}"
            print(error_message)
            return error_message

        try:
            response_data = response.json()
            # Assuming valid response contains pipelineResponse field
            audio_content = response_data["pipelineResponse"][0]["audio"][0]["audioContent"]
        except (KeyError, json.JSONDecodeError) as e:
            print("Error parsing TTS response:", e)
            return "Error: Unable to parse TTS response."

        # Process audio_content as needed (e.g., save as wav, play audio, etc.)
        return audio_content
   
    def tts(self,content,source_language):
       
        headers = {
            "Authorization" : self.authorization_key,
            "Content-Type": "application/json",
        }
 
        payload= {"pipelineTasks": [      
                        {
                            "taskType": "tts",
                            "config": {
                                        "language": {
                                                    "sourceLanguage": source_language
                                                    },
                                        "serviceId": self.tts_service_id[source_language],
                                        "gender": "female"
                                        }
                        }
                    ],
                    "inputData": {
                        "input": [
                            {
                                "source": content
                            }
                        ]
                    }
                    }
 
        payload_json= json.dumps(payload)
 
        response = requests.post(self.url, headers=headers, data=payload_json)
        print(f"raw response = {response}")
        if response.status_code ==200:
            response=response.json()
            return response["pipelineResponse"][0]["audio"][0]["audioContent"]
   
        else:
            print(f"Couldn't convert text to speech!!! - {response.status_code}")
            return None