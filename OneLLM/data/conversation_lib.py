import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    strategy: str = None

    skip_next: bool = False

    def get_prompt(self, prompt=None):
        if self.sep_style == SeparatorStyle.SINGLE:
            if prompt is None:
                ret = self.system + '\n\n' + self.sep
            else:
                ret = self.system + '\n\n' + prompt + '\n\n' + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + '\n' + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((224, 224))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    # image = image.resize((224, 224))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace('<image>', img_str)
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

    #Missing Prompt inside Combining Conversational Template
    def get_miss_prompt(self, modal: List[str], 
                        task_modals: List[str], 
                        use_text_modality:bool = False, 
                        use_table_modality:bool = False,
                        compensation_strategy:bool = True):
        status = "Input Modality Status: "
        #audio-video tasks
        if len(task_modals)==2 and 'video' in task_modals and 'audio' in task_modals:
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Missing; "
            if 'audio' in modal:
                status += "Audio: Present; "
            else:
                status += "Audio: Missing; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'

            if compensation_strategy:
                if 'video' in modal and 'audio' not in modal: #only-video
                    prompt += "The audio is missing. The assistant must use visual data to infer a probable audio context."
                elif 'audio' in modal and 'video' not in modal: #only-audio
                    prompt += "The video is missing. The assistant must use audio data to infer a probable visual context."
                elif 'audio' in modal and 'video' in modal: #complete
                    prompt += "Both video and audio are present."
                else:
                    raise NotImplementedError
        
        #image-text tasks
        if len(task_modals)==2 and 'image' in task_modals and 'text' in task_modals:
            if 'image' in modal:
                status += "Image: Present; "
            else:
                status += "Image: Missing; "
            if use_text_modality:
                status += "Textual description of the image: Present; "
            else:
                status += "Textual description of the image: Missing; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'

            if compensation_strategy:
                if 'image' in modal and not use_text_modality: #only-image
                    prompt += "The textual description of the image is missing. The assistant must use the image to infer a probable textual description context."
                elif use_text_modality and 'image' not in modal: #only-text
                    prompt += "The image is missing. The assistant must use the textual description of the image to infer a probable image context."
                elif 'image' in modal and use_text_modality: #complete
                    prompt += "Both the image and the textual description of the image are present."
                else:
                    raise NotImplementedError

        #audio-video-text tasks
        if len(task_modals)==3 and 'video' in task_modals and 'audio' in task_modals and 'text' in task_modals:
            if 'audio' in modal:
                status += "Audio: Present; "
            else:
                status += "Audio: Missing; "
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Missing; "
            if use_text_modality:
                status += "Input text: Present; "
            else:
                status += "Input text: Missing; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'
            if compensation_strategy:
                #one missing modality
                if 'video' in modal and 'audio' in modal and not use_text_modality: #audio+video
                    prompt += "The input text is missing. The assistant must use audio and visual data to infer a probable textual context."
                elif 'video' in modal and 'audio' not in modal and use_text_modality: #video+text
                    prompt += "The audio is missing. The assistant must use visual and textual data to infer a probable audio context."
                elif 'audio' in modal and 'video' not in modal and use_text_modality: #audio+text
                    prompt += "The video is missing. The assistant must use audio and textual data to infer a probable video context."
                #two missing modalities
                elif 'video' in modal and 'audio' not in modal and not use_text_modality: #video
                    prompt += "The audio and input text are missing. The assistant must use visual data to infer a probable audio and textual context."
                elif use_text_modality and 'video' not in modal and 'audio' not in modal: #text
                    prompt += "The video and audio are missing. The assistant must use textual data to infer a probable visual and audio context."
                elif 'audio' in modal and 'video' not in modal and not use_text_modality: #audio
                    prompt += "The video and input text are missing. The assistant must use audio to infer a probable video and textual context."
                #complete
                elif 'video' in modal and 'audio' in modal and use_text_modality: #video-audio-text
                    prompt += "Audio, visual and textual data are all present."
                else:
                    raise NotImplementedError


        #image-video-tabular-text 
        if len(task_modals)==4 and 'video' in task_modals and 'image' in task_modals and 'text' in task_modals and 'table' in task_modals:
            if 'image' in modal:
                status += "Image: Present; "
            else:
                status += "Image: Missing; "
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Missing; "
            if use_table_modality:
                status += "Tabular information: Present; "
            else:
                status += "Tabular information: Missing; "
            if use_text_modality:
                status += "Input text: Present; "
            else:
                status += "Input text: Missing; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'
            if compensation_strategy:
                #one missing modality
                if 'video' in modal and 'image' in modal and use_table_modality and not use_text_modality: #image+video+table
                    prompt += "The input text is missing. The assistant must use video, image and tabular data to infer a probable textual context."
                elif 'video' in modal and 'image' in modal and not use_table_modality and use_text_modality: #image+video+text
                    prompt += "The tabular data are missing. The assistant must use video, image and textual data to infer a probable tabular context."
                elif 'video' in modal and 'image' not in modal and use_table_modality and use_text_modality: #video+text+table
                    prompt += "The image is missing. The assistant must use video, tabular and textual data to infer a probable image context."
                elif 'image' in modal and 'video' not in modal and use_table_modality and use_text_modality: #image+text+table
                    prompt += "The video is missing. The assistant must use image, tabular and textual data to infer a probable video context."
                #two missing modalities
                elif 'video' in modal and 'image' not in modal and not use_text_modality and use_table_modality: #video+table
                    prompt += "The image and input text are missing. The assistant must use video and tabular data to infer a probable image and textual context."
                elif use_text_modality and use_table_modality and 'video' not in modal and 'image' not in modal: #text+table
                    prompt += "The video and image are missing. The assistant must use tabular and textual data to infer a probable video and image context."
                elif 'image' in modal and 'video' not in modal and not use_text_modality and use_table_modality: #image+table
                    prompt += "The video and input text are missing. The assistant must use image and tabular data to infer a probable video and textual context."
                elif 'image' in modal and 'video' in modal and not use_text_modality and not use_table_modality: #image+video
                    prompt += "The tabular and textual data are missing. The assistant must use video and image data to infer a probable tabular and textual context."
                elif 'image' in modal and 'video' not in modal and use_text_modality and not use_table_modality: #image+text
                    prompt += "The video and tabular data are missing. The assistant must use image and textual data to infer a probable video and tabular context."
                elif 'image' not in modal and 'video' in modal and use_text_modality and not use_table_modality: #video+text
                    prompt += "The image and tabular data are missing. The assistant must use video and textual data to infer a probable image and tabular context."
                #three missing modalities
                elif 'video' in modal and 'image' not in modal and not use_table_modality and not use_text_modality: #video
                    prompt += "The image, tabular and textual data are missing. The assistant must use video data to infer a probable image, textual and tabular context."
                elif 'video' not in modal and 'image' in modal and not use_table_modality and not use_text_modality: #image
                    prompt += "The video, tabular and textual data are missing. The assistant must use image data to infer a probable video, tabular and textual context."
                elif 'video' not in modal and 'image' not in modal and use_table_modality and not use_text_modality: #table
                    prompt += "The video, image and textual data are missing. The assistant must use tabular data to infer a probable video, image and textual context."
                elif 'image' not in modal and 'video' not in modal and not use_table_modality and use_text_modality: #text
                    prompt += "The video, image and tabular data are missing. The assistant must use textual data to infer a probable video, image and tabular context."
                #complete
                elif 'video' in modal and 'image' in modal and use_text_modality and use_table_modality: #video-audio-text-table
                    prompt += "Image, video, tabular and textual data are all present."
                else:
                    raise NotImplementedError
        return prompt

    #Missing Prompt inside Original Conversational Template
    def get_miss_prompt2(self, modal: List[str], task_modals: List[str], use_text_modality: bool = False):
        status = "Input Modality Status: "
        #audio-video tasks
        if len(task_modals)==2 and 'video' in task_modals and 'audio' in task_modals:
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Missing; "
            if 'audio' in modal:
                status += "Audio: Present; "
            else:
                status += "Audio: Missing; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'

            if 'video' in modal and 'audio' not in modal: #only-video
                prompt += "The assistant uses only visual data to answer the questions."
            elif 'audio' in modal and 'video' not in modal: #only-audio
                prompt += "The assistant uses only audio data to answer the questions."
            elif 'audio' in modal and 'video' in modal: #complete
                prompt += "The assistant combines visual and audio data to answer the questions."
            else:
                raise NotImplementedError
        
        #image-text tasks
        if len(task_modals)==2 and 'image' in task_modals and 'text' in task_modals:
            if 'image' in modal:
                status += "Image: Present; "
            else:
                status += "Image: Missing; "
            if use_text_modality:
                status += "Textual description of the image: Present; "
            else:
                status += "Textual description of the image: Missing; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'

            if 'image' in modal and not use_text_modality: #only-image
                prompt += "The assistant uses only visual data to answer the questions."
            elif use_text_modality and 'image' not in modal: #only-text
                prompt += "The assistant uses only the textual description of the image to answer the questions."
            elif 'image' in modal and use_text_modality: #complete
                prompt += "The assistant combines visual data and the textual description of the image to answer the questions."
            else:
                raise NotImplementedError

        #audio-video-text tasks
        if len(task_modals)==3 and 'video' in task_modals and 'audio' in task_modals and 'text' in task_modals:
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Missing; "
            if 'audio' in modal:
                status += "Audio: Present; "
            else:
                status += "Audio: Missing; "
            if use_text_modality:
                status += "Input text: Present; "
            else:
                status += "Input text: Missing; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'

            if 'video' in modal and 'audio' in modal and not use_text_modality: #audio+video
                prompt += "The assistant uses visual and audio data to answer the questions."
            elif 'video' in modal and 'audio' not in modal and use_text_modality: #video+text
                prompt += "The assistant uses visual and textual data to answer the questions."
            elif 'audio' in modal and 'video' not in modal and use_text_modality: #audio+text
                prompt += "The assistant uses audio and textual data to answer the questions."
            elif 'video' in modal and 'audio' in modal and use_text_modality: #video-audio-text
                prompt += "The assistant combines visual, audio and textual data to answer the questions."
            else:
                raise NotImplementedError
        return prompt
    
    #Missing Prompt inside the last Human Instruction
    def get_miss_prompt_inside(self, prompt:str, modal: List[str], task_modals: List[str], use_text_modality:bool = False):
        status = "Input Modality Status: "
        prompt = prompt.lower()

        #audio-video tasks
        if len(task_modals)==2 and 'video' in task_modals and 'audio' in task_modals:
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Missing; "
            if 'audio' in modal:
                status += "Audio: Present; "
            else:
                status += "Audio: Missing; "
            status = status[:-2] + '.'

            if 'video' in modal and 'audio' not in modal: #only-video
                prompt_inside = status + "\nThe audio is missing. Use visual data to infer a probable audio context and " + prompt
            elif 'audio' in modal and 'video' not in modal: #only-audio
                prompt_inside = status + "\nThe video is missing. Use audio data to infer a probable visual context and " + prompt
            elif 'audio' in modal and 'video' in modal: #complete
                prompt_inside = status + "\nBoth video and audio are present, " + prompt
            else:
                raise NotImplementedError

        return prompt_inside
    
    #Prototipe Prompt inside Combining Conversational Template
    def get_prototipe_prompt(self, modal: List[str], task_modals: List[str], use_text_modality:bool = False, use_table_modality:bool = False, input_text: bool = False, input_table: bool = False):
        """
        use_text_modality: whether textual modality is present or not
        input_text: whether textual modality is among the task modalities or not
        """
        status = "Input Modality Status: "
        #audio-video tasks
        if len(task_modals)==2 and 'video' in task_modals and 'audio' in task_modals and not input_text:
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Prototype; "
            if 'audio' in modal:
                status += "Audio: Present; "
            else:
                status += "Audio: Prototype; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'

            if 'video' in modal and 'audio' not in modal: #only-video
                prompt += "The assistant receives an approximate audio data and uses it to generate a response as accurately as possible."
            elif 'audio' in modal and 'video' not in modal: #only-audio
                prompt += "The assistant receives an approximate visual data and uses it to generate a response as accurately as possible."
            elif 'audio' in modal and 'video' in modal: #complete
                prompt += "Both video and audio are present."
            else:
                raise NotImplementedError
        
        #image-text tasks
        if len(task_modals)==1 and 'image' in task_modals and input_text:
            if 'image' in modal:
                status += "Image: Present; "
            else:
                status += "Image: Prototype; "
            if use_text_modality:
                status += "Textual description of the image: Present; "
            else:
                status += "Textual description of the image: Prototype; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'

            if 'image' in modal and not use_text_modality: #only-image
                prompt += "The assistant receives an approximate description of the image and uses it to generate a response as accurately as possible."
            elif use_text_modality and 'image' not in modal: #only-text
                prompt += "The assistant receives an approximate image and uses it to generate a response as accurately as possible."
            elif 'image' in modal and use_text_modality: #complete
                prompt += "Both the image and the textual description of the image are present."
            else:
                raise NotImplementedError

        #audio-video-text tasks
        if len(task_modals)==2 and 'video' in task_modals and 'audio' in task_modals and input_text:
            if 'audio' in modal:
                status += "Audio: Present; "
            else:
                status += "Audio: Prototype; "
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Prototype; "
            if use_text_modality:
                status += "Input text: Present; "
            else:
                status += "Input text: Prototype; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'
            #one missing modality
            if 'video' in modal and 'audio' in modal and not use_text_modality: #audio+video
                prompt += "The assistant receives an approximate input text data and uses it to generate a response as accurately as possible."
            elif 'video' in modal and 'audio' not in modal and use_text_modality: #video+text
                prompt += "The assistant receives an approximate audio data and uses it to generate a response as accurately as possible."
            elif 'audio' in modal and 'video' not in modal and use_text_modality: #audio+text
                prompt += "The assistant receives an approximate visual data and uses it to generate a response as accurately as possible."
            #two missing modalities
            elif 'video' in modal and 'audio' not in modal and not use_text_modality: #video
                prompt += "The assistant receives an approximate audio and input text data and uses them to generate a response as accurately as possible."
            elif 'audio' in modal and 'video' not in modal and not use_text_modality: #audio
                prompt += "The assistant receives an approximate visual and input text data and uses them to generate a response as accurately as possible."
            elif use_text_modality and 'video' not in modal and 'audio' not in modal: #text
                prompt += "The assistant receives an approximate visual and audio data and uses them to generate a response as accurately as possible."
            # complete
            elif 'video' in modal and 'audio' in modal and use_text_modality: #video-audio-text
                prompt += "Audio, visual and textual data are all present."
            else:
                raise NotImplementedError
        
        #image-video-tabular-text 
        if len(task_modals)==2 and 'video' in task_modals and 'image' in task_modals and input_text and input_table:
            if 'image' in modal:
                status += "Image: Present; "
            else:
                status += "Image: Prototype; "
            if 'video' in modal:
                status += "Video: Present; "
            else:
                status += "Video: Prototype; "
            if use_table_modality:
                status += "Tabular information: Present; "
            else:
                status += "Tabular information: Prototype; "
            if use_text_modality:
                status += "Input text: Present; "
            else:
                status += "Input text: Prototype; "
            status = status[:-2] + '.'
            prompt = status + '\n\n'
            #one missing modality
            if 'video' in modal and 'image' in modal and use_table_modality and not use_text_modality: #image+video+table
                prompt += "The assistant receives an approximate input text and uses it to generate a response as accurately as possible."
            elif 'video' in modal and 'image' in modal and not use_table_modality and use_text_modality: #image+video+text
                prompt += "The assistant receives an approximate tabular data and uses it to generate a response as accurately as possible."
            elif 'video' in modal and 'image' not in modal and use_table_modality and use_text_modality: #video+text+table
                prompt += "The assistant receives an approximate image and uses it to generate a response as accurately as possible."
            elif 'image' in modal and 'video' not in modal and use_table_modality and use_text_modality: #image+text+table
                prompt += "The assistant receives an approximate video and uses it to generate a response as accurately as possible."
            #two missing modalities
            elif 'video' in modal and 'image' not in modal and not use_text_modality and use_table_modality: #video+table
                prompt += "The assistant receives approximate image and input text data and uses them to generate a response as accurately as possible." 
            elif use_text_modality and use_table_modality and 'video' not in modal and 'image' not in modal: #text+table
                prompt += "The assistant receives approximate video and image data and uses them to generate a response as accurately as possible." 
            elif 'image' in modal and 'video' not in modal and not use_text_modality and use_table_modality: #image+table
                prompt += "The assistant receives approximate video and input text data and uses them to generate a response as accurately as possible." 
            elif 'image' in modal and 'video' in modal and not use_text_modality and not use_table_modality: #image+video
                prompt += "The assistant receives approximate tabular and textual data and uses them to generate a response as accurately as possible." 
            elif 'image' in modal and 'video' not in modal and use_text_modality and not use_table_modality: #image+text
                prompt += "The assistant receives approximate video and tabular data and uses them to generate a response as accurately as possible." 
            elif 'image' not in modal and 'video' in modal and use_text_modality and not use_table_modality: #video+text
                prompt += "The assistant receives approximate image and tabular data and uses them to generate a response as accurately as possible." 
            #three missing modalities
            elif 'video' in modal and 'image' not in modal and not use_table_modality and not use_text_modality: #video
                prompt += "The assistant receives approximate image, tabular and textual data and uses them to generate a response as accurately as possible." 
            elif 'video' not in modal and 'image' in modal and not use_table_modality and not use_text_modality: #image
                prompt += "The assistant receives approximate video, tabular and textual data and uses them to generate a response as accurately as possible." 
            elif 'video' not in modal and 'image' not in modal and use_table_modality and not use_text_modality: #table
                prompt += "The assistant receives approximate video, image and textual data and uses them to generate a response as accurately as possible." 
            elif 'image' not in modal and 'video' not in modal and not use_table_modality and use_text_modality: #text
                prompt += "The assistant receives approximate video, image and tabular data and uses them to generate a response as accurately as possible." 
            #complete
            elif 'video' in modal and 'image' in modal and use_text_modality and use_table_modality: #video-audio-text-table
                prompt += "Image, video, tabular and textual data are all present."
            else:
                raise NotImplementedError
        return prompt
    
conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Give three tips for staying healthy."),
        ("Assistant",
         "Sure, here are three tips for staying healthy:\n"
         "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
         "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
         "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
         "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
         "activities at least two days per week.\n"
         "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
         "vegetables, whole grains, lean proteins, and healthy fats can help support "
         "your overall health. Try to limit your intake of processed and high-sugar foods, "
         "and aim to drink plenty of water throughout the day.\n"
         "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
         "and mental health. Adults should aim for seven to nine hours of sleep per night. "
         "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
         "help improve the quality of your sleep.")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(),

    # (
    #     ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
    #     ("Assistant",
    #         "Renewable energy sources are those that can be replenished naturally in a relatively "
    #         "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
    #         "Non-renewable energy sources, on the other hand, are finite and will eventually be "
    #         "depleted, such as coal, oil, and natural gas. Here are some key differences between "
    #         "renewable and non-renewable energy sources:\n"
    #         "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
    #         "energy sources are finite and will eventually run out.\n"
    #         "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
    #         "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
    #         "and other negative effects.\n"
    #         "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
    #         "have lower operational costs than non-renewable sources.\n"
    #         "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
    #         "locations than non-renewable sources.\n"
    #         "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
    #         "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
    #         "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
    #         "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    # )
    offset = 2,
    sep_style = SeparatorStyle.SINGLE,
    sep = "###",
    )

conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
- You are a helpful language and vision assistant.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mpt_text = Conversation(
    system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

simple_conv = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there! How can I help you today?")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_multimodal = Conversation(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
           "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you today?\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_mpt_multimodal = Conversation(
    system="""<|im_start|>system
- You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

simple_conv_legacy = Conversation(
    system="You are LLaVA, a large language model trained by UW Madison WAIV Lab."
           "You are designed to assist human with a variety of tasks using natural language."
           "Follow the instructions carefully.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!\n\n### Response:"),
        ("Assistant", "Hi there!  How can I help you today?\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v1 = Conversation(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
           "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

food101_1 = Conversation(
    system= "You are an expert assistant specialized in food image classification."
            "Your task is to analyze images from the Food101 dataset and provide detailed descriptions and accurate class labels from the 101 specific food categories."
            "Ensure that your classifications are precise, avoiding general terms like 'dessert' or 'food' unless absolutely necessary."
            "Provide clear and detailed explanations for your classifications.",
    roles=("Human", "Assistant"),
    messages=(),
    offset = 2,
    sep_style = SeparatorStyle.SINGLE,
    sep = "###",
    )

conv_v1_2_audio_video = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions, combining visual and audio data.", 
    roles=("Human", "Assistant"),
    messages=(),
    offset = 2,
    sep_style = SeparatorStyle.SINGLE,
    sep = "###",
    )

conv_v1_2_image_text = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions, combining image data and textual description of the image.", 
    roles=("Human", "Assistant"),
    messages=(),
    offset = 2,
    sep_style = SeparatorStyle.SINGLE,
    sep = "###",
    )

conv_v1_2_audio_video_text = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions, combining audio, visual and textual data.", 
    roles=("Human", "Assistant"),
    messages=(),
    offset = 2,
    sep_style = SeparatorStyle.SINGLE,
    sep = "###",
    )

conv_v1_2_image_video_text_table = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions, combining video, image, tabular and textual data.", 
    roles=("Human", "Assistant"),
    messages=(),
    offset = 2,
    sep_style = SeparatorStyle.SINGLE,
    sep = "###",
    )

default_conversation = conv_v1_2
conv_templates = {
    "default": conv_v1_2,
    "simple": simple_conv,
    "simple_legacy": simple_conv_legacy,
    "multimodal": simple_conv_multimodal,
    "mpt_multimodal": simple_conv_mpt_multimodal,
    "llava_v1": conv_llava_v1,

    # fastchat
    "v1": conv_v1_2,
    "v1_audio_video": conv_v1_2_audio_video,
    "v1_image_text": conv_v1_2_image_text,
    "v1_audio_video_text": conv_v1_2_audio_video_text,
    "v1_image_video_text_table": conv_v1_2_image_video_text_table,
    "food101_1": food101_1,
    "bair_v1": conv_bair_v1,
    "vicuna_v1_1": conv_vicuna_v1_1,
    "mpt": conv_mpt,
    "mpt_text": conv_mpt_text,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
