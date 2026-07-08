from __future__ import annotations

EXAMPLE_TEXTS = {
    "en": {
        "Albert Einstein": "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc², which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.",
        "AAAI": "The Association for the Advancement of Artificial Intelligence (AAAI) is an international scientific society devoted to promote research in, and responsible use of, artificial intelligence (AI). AAAI also aims to increase public understanding of AI, improve the teaching and training of AI practitioners, and provide guidance for research planners and funders concerning the importance and potential of current AI developments and future directions.",
        "Singapore": "Singapore, officially the Republic of Singapore, is a sovereign country as well as a city-state. It is nicknamed as 'The Lion City', 'The Garden City' or 'The Little Red Dot'. It is an island state at the southern end of the Malay Peninsula in Southeast Asia, between the Straits of Malacca and the South China Sea. Singapore is about one degree of latitude (137 kilometres or 85 miles) north of the equator. About 5.70 million people live in Singapore. About 3.31 million are citizens. Most of them are ethnically Chinese, Malay, or Indian, as well as a smaller number of other Asians and Europeans.",
        "AAAI-2026": "In 2026, the AAAI Conference on Artificial Intelligence was held in Singapore, bringing together researchers and practitioners from academia, industry, and government. The conference featured peer-reviewed technical papers, invited talks, workshops, tutorials, and poster sessions covering a broad range of topics in artificial intelligence. Singapore served as the host location for the event, providing conference facilities and infrastructure to support international participation. The 2026 edition continued AAAI’s annual conference series and contributed to the dissemination of current research results and ongoing developments in the field of artificial intelligence.",
        "TP53": "p53, also known as tumor protein p53 (TP53), is a regulatory transcription factor protein that is often mutated in human cancers. p53 has been described as 'the guardian of the genome' because of its role in conserving stability by preventing genome mutation. Hence TP53 is classified as a tumor suppressor gene. The TP53 gene is the most frequently mutated gene (>50%) in human cancer, indicating that the TP53 gene plays a crucial role in preventing cancer formation. TP53 gene encodes proteins that bind to DNA and regulate gene expression to prevent mutations of the genome.",
        "p21": "p21Cip1 (alternatively p21Waf1), also known as cyclin-dependent kinase inhibitor 1 or CDK-interacting protein 1, is a cyclin-dependent kinase inhibitor (CKI) that is capable of inhibiting all cyclin/CDK complexes. p21 represents a major target of p53 activity and thus is associated with linking DNA damage to cell cycle arrest. This protein is encoded by the CDKN1A gene located on chromosome 6 (6p21.2) in humans.",
    },
    "ru": {
        "Юрий Гагарин": (
            "Юрий Алексеевич Гагарин (9 марта 1934, Клушино — 27 марта 1968, "
            "село Новосёлово, Владимирская область) — советский космонавт и "
            "военный лётчик, первый человек, совершивший космический полёт. "
            "Герой Советского Союза, кавалер высших знаков отличия ряда государств, "
            "почётный гражданин многих российских и зарубежных городов."
        ),
        "Алексей Леонов": (
            "Алексей Архипович Леонов (30 мая 1934, Листвянка, Западно-Сибирский "
            "край — 11 октября 2019, Басманный район, Москва) — лётчик-космонавт "
            "СССР № 11, первый человек в мире, вышедший в открытый космос. "
            "Дважды Герой Советского Союза (1965, 1975), генерал-майор авиации "
            "(1975), лауреат Государственной премии СССР (1981)."
        ),
    },
}

PERSONAL_SEARCH_PROMPTS = {
    "en": (
        "Find and extract fresh, up-to-date information from the internet about "
        "{person}, and return one paragraph summarizing it. Return only the "
        "paragraph, with no other text."
    ),
    "ru": (
        "Найдите и извлеките из интернета свежую и актуальную информацию о "
        "{person} и верните параграф, который суммирует эту информацию. "
        "Верните только параграф, никакого другого текста."
    ),
}


def get_example_texts(language: str) -> dict[str, str]:
    return EXAMPLE_TEXTS.get(language, EXAMPLE_TEXTS["en"])


def personal_search_prompt(language: str, person: str) -> str:
    template = PERSONAL_SEARCH_PROMPTS.get(language, PERSONAL_SEARCH_PROMPTS["en"])
    return template.format(person=person)
