__all__ = ["app"]

import gradio as gr
from binoculars import Binoculars3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import PyPDF2
from fixthaipdf import clean

model = AutoModelForCausalLM.from_pretrained("scb10x/llama-3-typhoon-v1.5-8b", device_map="auto", load_in_8bit=True)
model1 = AutoModelForCausalLM.from_pretrained("scb10x/llama-3-typhoon-v1.5x-8b-instruct", device_map="auto", load_in_8bit=True)

BINO = Binoculars3(observer_name_or_path=model, performer_name_or_path=model1, name="scb10x/llama-3-typhoon-v1.5-8b")
TOKENIZER = BINO.tokenizer
MINIMUM_TOKENS = 64
BINOCULARS_ACCURACY_THRESHOLD_TH = 0.9579831932773109  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD_TH = 0.7563025210084033  # optimized for low-fpr

def count_tokens(text):
    return len(TOKENIZER(text).input_ids)


def run_detector(input_str):
    if count_tokens(input_str) < MINIMUM_TOKENS:
        gr.Warning(f"Too short length. Need minimum {MINIMUM_TOKENS} tokens to run Binoculars.")
        return ""
    return f"{BINO.predict(input_str)}"


def change_mode(mode):
    if mode == "Low False Positive Rate":
        BINO.set_threshold(BINOCULARS_FPR_THRESHOLD_TH)
    elif mode == "High Accuracy":
        BINO.set_threshold(BINOCULARS_ACCURACY_THRESHOLD_TH)
    else:
        gr.Error(f"Invalid mode selected.")
    return mode

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open (pdf_path, "rb") as file:
        pdf = PyPDF2.PdfReader(file)
        text = ""
        text='\n'.join([i.extract_text() for i in pdf.pages])
        text = clean(text)
    return text

def handle_pdf_upload(pdf_file):
    if pdf_file is not None:
        return extract_text_from_pdf(pdf_file.name)
    return ""

# def load_set(progress=gr.Progress()):
#     tokens = [None] * 24
#     for count in progress.tqdm(tokens, desc="Counting Tokens..."):
#         time.sleep(0.01)
#     return ["Loaded"] * 2


css = """
.green { color: black!important;line-height:1.9em; padding: 0.2em 0.2em; background: #ccffcc; border-radius:0.5rem;}
.red { color: black!important;line-height:1.9em; padding: 0.2em 0.2em; background: #ffad99; border-radius:0.5rem;}
.hyperlinks {
  display: flex;
  align-items: center;
  align-content: center;
  padding-top: 12px;
  justify-content: flex-end;
  margin: 0 10px; /* Adjust the margin as needed */
  text-decoration: none;
  color: #000; /* Set the desired text color */
}
"""

# Most likely human generated, #most likely AI written

capybara_problem = '''โลกแห่งดาราศาสตร์นั้นช่างงดงามและเต็มไปด้วยความลับอันน่าค้นหา แต่ทว่าคาปิบาราตัวหนึ่งกลับเลือกที่จะก้าวเข้าสู่ความลึกลับนี้ด้วยความกล้าหาญ ในฐานะนักดาราศาสตร์หนุ่มผู้เปี่ยมไปด้วยความมุ่งมั่น มันทุ่มเทเวลาศึกษาค้นคว้าเกี่ยวกับกาแล็กซีอันไกลโพ้นและวัฏจักรชีวิตของดวงดาวอย่างถี่ถ้วน ทั้งกลางวันและกลางคืน มันเฝ้ามองท้องฟ้าผ่านกล้องโทรทรรศน์ วิเคราะห์ข้อมูล และค้นหาข้อเท็จจริงใหม่ ๆ เกี่ยวกับเอกภพอันกว้างใหญ่ไพศาล

ความหลงใหลในดวงดาวของคาปิบารานั้นเปรียบเสมือนเปลวไฟที่ลุกโชน มันทุ่มเทแรงกายแรงใจเพื่อไขปริศนาของจักรวาล ค้นหาคำตอบต่อคำถามอันเก่าแก่เกี่ยวกับจุดกำเนิดและจุดจบของดวงดาว และพยายามทำความเข้าใจถึงความกว้างใหญ่ไพศาลของอวกาศ

แม้เส้นทางการเป็นนักดาราศาสตร์จะเต็มไปด้วยอุปสรรคและความท้าทาย แต่คาปิบาราก็ไม่เคยท้อถอย มันเรียนรู้ทักษะใหม่ ๆ อยู่เสมอ พัฒนาความรู้ด้านฟิสิกส์ ดาราศาสตร์ และคณิตศาสตร์อย่างต่อเนื่อง และมุ่งมั่นที่จะไขความลับของจักรวาลให้กระจ่าง

ด้วยความมุ่งมั่นและความทุ่มเทของคาปิบารา มันจึงกลายเป็นแรงบันดาลใจให้กับนักดาราศาสตร์รุ่นใหม่มากมาย หลาย ๆ คนต่างยกย่องมันในฐานะนักค้นคว้าผู้เปี่ยมไปด้วยความกล้าหาญและความมุ่งมั่นที่จะไขปริศนาของจักรวาล

เรื่องราวของคาปิบารานักดาราศาสตร์นี้สอนให้เรารู้ว่า ไม่ว่าเราจะเป็นใครก็ตาม ล้วนมีโอกาสที่จะทำตามความฝันและไขความลับของจักรวาลได้ เพียงแค่ต้องมีความกล้าหาญ มุ่งมั่น และทุ่มเทอย่างเต็มที่'''

with gr.Blocks(css=css,
               theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"])) as app:
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<p><h1> binoculars-TH: zero-shot llm-text detector for Thai</h1>")
        with gr.Column(scale=1):
            gr.HTML("""
            <p>
            <a href="https://arxiv.org/abs/2401.12070" target="_blank">paper</a>
                
            <a href="https://github.com/Saltywan/Binoculars/" target="_blank">code</a>
                
            <a href="mailto:thanapolwan79@gmail.com" target="_blank">contact</a>
            """, elem_classes="hyperlinks")
    with gr.Row():
        input_box = gr.Textbox(value=capybara_problem, placeholder="Enter text here", lines=8, label="Input Text")
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
    with gr.Row():
        # dropdown option for mode
        dropdown_mode = gr.Dropdown(["Low False Positive Rate", "High Accuracy"],
                                    label="Mode",
                                    show_label=True,
                                    value="High Accuracy"
                                    )
        submit_button = gr.Button("Run Binoculars", variant="primary")
        clear_button = gr.ClearButton()
    with gr.Row():
        output_text = gr.Textbox(label="Prediction", value="Most likely AI-Generated")

    with gr.Row():
        gr.HTML("<p><p><p>")
    with gr.Row():
        gr.HTML("<p><p><p>")
    with gr.Row():
        gr.HTML("<p><p><p>")

    with gr.Accordion("Disclaimer", open=False):
        gr.Markdown(
            """
            - `Accuracy` :
                - AI-generated text detectors aim for accuracy, but no detector is perfect.
                - If you choose "high accuracy" mode, then the threshold between human and machine is chosen to maximize the F1 score on our validation dataset.
                - If you choose the "low false-positive rate" mode, the threshold for declaring something to be AI generated will be set so that the false positive (human text wrongly flagged as AI) rate is below 0.01% on our validation set. 
                - The provided prediction is for demonstration purposes only. This is not offered as a consumer product.
                - Users are advised to exercise discretion, and we assume no liability for any use.
            - `Recommended detection Use Cases` : 
                - In this work, our focus is on achieving a low false positive rate, crucial for sensitive downstream use cases where false accusations are highly undesireable. 
                - The main focus of our research is on content moderation, e.g., detecting AI-generated reviews on Amazon/Yelp, detecting AI generated social media posts and news, etc. We feel this application space is most compelling, as LLM detection tools are best used by professionals in conjunction with a broader set of moderation tools and policies. 
            - `Known weaknesses` :
                - As noted in our paper, Binoculars exhibits superior detection performance in the English language compared to other languages.  Non-English text makes it more likely that results will default to "human written." 
                - Binoculars considers verbatim memorized texts to be "AI generated." For example, most language models have memorized and can recite the US constitution. For this reason, text from the constitution, or other highly memorized sources, may be classified as AI written. 
                - We recommend using 200-300 words of text at a time. Fewer words make detection difficult, as can using more than 1000 words. Binoculars will be more likely to default to the "human written" category if too few tokens are provided.
            """
        )

    with gr.Accordion("Original paper", open=False):
        gr.Markdown(
            """
            ```bibtex
                @misc{hans2024spotting,
                      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text}, 
                      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
                      year={2024},
                      eprint={2401.12070},
                      archivePrefix={arXiv},
                      primaryClass={cs.CL}
                }
            """
        )

    # confidence_bar = gr.Label(value={"Confidence": 0})

    # clear_button.click(lambda x: input_box., )
    submit_button.click(run_detector, inputs=input_box, outputs=output_text)
    clear_button.click(lambda: ("", ""), outputs=[input_box, output_text])
    dropdown_mode.change(change_mode, inputs=[dropdown_mode], outputs=[dropdown_mode])
    pdf_upload.change(handle_pdf_upload, inputs=pdf_upload, outputs=input_box)