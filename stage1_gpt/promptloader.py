import sys

# can add more prompts
def get_general_prompt(prompt_idx):
    if prompt_idx == 0:
        general_prompt = (
                        "Please describe the {video_type} clip in the following four steps: "
                        "1. Identify main characters (if {label_type} are available){char_text}; "
                        "2. Describe the actions of characters in one sentence, i.e., who is doing what, focusing on the movements; " 
                        "3. Describe the interactions between characters in one sentence, such as looking; "
                        "4. Describe the facial expressions of characters in one sentence. "
                        "Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description. "   
                        "Make sure you do not hallucinate information. "
                        "###ANSWER TEMPLATE###: 1. Main characters: ''; 2. Actions: ''; 3. Character-character interactions: ''; 4. Facial expressions: ''."
        ) 
    else:
        print("The prompt idx does not exist")
        sys.exit()
    
    return general_prompt