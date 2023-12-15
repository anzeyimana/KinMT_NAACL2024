from typing import List

from google.cloud import translate
def init_google_translate_client():
    client = translate.TranslationServiceClient()
    return client

def gogle_translate(client, text_list: List[str], src_lang='rw', tgt_lang='en') -> List[str]:
    location = "global"
    project_id = 'morpho-234417'
    parent = f"projects/{project_id}/locations/{location}"
    response = client.translate_text(
             request={
                 "parent": parent,
                 "contents": text_list,
                 "mime_type": "text/plain",  # mime types: text/plain, text/html
                 "source_language_code": src_lang,
                 "target_language_code": tgt_lang,
             }
         )
    translations = [translation.translated_text for translation in response.translations]
    return translations
