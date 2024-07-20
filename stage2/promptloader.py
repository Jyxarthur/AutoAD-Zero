# can add more prompt options
def get_user_prompt(prompt_idx, verb_list, text_pred, duration):
    if prompt_idx == 0:
        user_prompt = (
            "Please summarise the following description for one TV series clip into ONE succinct audio description (AD) sentence.\n"
            f"Description: {text_pred}\n\n"
            "Focus on the most attractive characters and their actions.\n"
            "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
            "For actions, avoid mentioning the camera, and do not focus on 'talking' or position-related ones such as 'sitting' and 'standing'.\n"
            "Do not mention characters' mood.\n"
            "Do not hallucinate information that is not mentioned in the input.\n"
            f"Try to identify the following motions (with decreasing priorities): {verb_list}, and use them in the description.\n"
            "Provide the AD from a narrator perspective and adjust the length of the output according to the duration.\n"
            f"Duration of the video clip: {duration}s\n\n"
            "For example, a response of duration 0.8s could be: {'summarised_AD': 'She looks at Riker.'}.\n"
            "Another example response of duration 1.4s is: {'summarised_AD': 'Paul looks at his wife lovingly.'}.\n"
            "An example response of duration 2.6s is: {'summarised_AD': 'He watches Tasha calmly battle with the figure.'}.\n"
        )

    elif prompt_idx == 1:
        user_prompt = (
            "Please summarise the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
            f"Description: {text_pred}\n\n"
            "Focus on the most attractive characters and their actions (focus on point 2., supplemented by point 3.).\n"
            "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
            "For actions, avoid mentioning the camera, and do not focus on 'talking' or position-related ones such as 'sitting' and 'standing'.\n"
            "Do not mention characters' mood.\n"
            "Do not hallucinate information that is not mentioned in the input.\n"
            f"Try to identify the following motions (with decreasing priorities): {verb_list}, and use them in the description.\n"
            "Provide the AD from a narrator perspective and adjust the length of the output according to the duration.\n"
            f"Duration of the video clip: {duration}s\n\n"
            "For example, a response of duration 0.8s could be: {'summarised_AD': 'She looks at Riker.'}.\n"
            "Another example response of duration 1.4s is: {'summarised_AD': 'Paul looks at his wife lovingly.'}.\n"
            "An example response of duration 2.6s is: {'summarised_AD': 'He watches Tasha calmly battle with the figure.'}.\n"
        )
    
    return user_prompt