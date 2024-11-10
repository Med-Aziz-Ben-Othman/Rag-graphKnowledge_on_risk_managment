import os
from time import sleep
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from Rag_System.data_prep.Data_prep import process_pdf
from Rag_System.data_prep.chuncking_data import generate_chunks_with_gpt4, refine_query_for_rag_system
from Rag_System.pinecone.Pinecone_init import setup_pinecone_index, upsert_documents_to_index, create_retrieval_chain

def save_output_to_file(output_folder, answer, context):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, "answer_with_context.txt")
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write("Answer with knowledge:\n")
        file.write(answer + "\n\n")
        file.write("Context used:\n")
        file.write(context)
    print(f"Output saved to {output_file_path}")

def main():
    pdf_path_pmi = "Projet_AI_Cognition/data/PMI_RM-standard.pdf"
    pdf_path_pmbok = "Projet_AI_Cognition/data/PMBOK6-2017.pdf"
    start_page_pmi, end_page_pmi = 13, 124
    start_page_pmbok, end_page_pmbok = 37, 571
    dictionary_path = "Projet_AI_Cognition/data/frequency_dictionary_en_82_765.txt"
    processed_text_path_pmi = "output_Rag_System/processed_data/processed_text_pmi.txt"
    processed_text_path_pmbok = "output_Rag_System/processed_data/processed_text_pmbok.txt"
    output_folder = "output_Rag_System"

    PMI_doc = process_pdf(pdf_path_pmi, start_page_pmi, end_page_pmi, dictionary_path, processed_text_path_pmi)
    PMBOK_doc = process_pdf(pdf_path_pmbok, start_page_pmbok, end_page_pmbok, dictionary_path, processed_text_path_pmbok)

    with open(processed_text_path_pmi, "r") as file:
        text_pmi = file.read()
    documents_pmi = generate_chunks_with_gpt4(text_pmi)

    with open(processed_text_path_pmbok, "r") as file:
        text_pmbok = file.read()
    documents_pmbok = generate_chunks_with_gpt4(text_pmbok)

    combined_chunked_documents = documents_pmi + documents_pmbok
    print(f"Created {len(combined_chunked_documents)} chunks.")
    sleep(2)

    # Pinecone Index setup and document upsert
    index = setup_pinecone_index()
    docsearch = upsert_documents_to_index(index, combined_chunked_documents)

    # Refine the query and get the answer
    query = """
        The project involves the construction of a state-of-the-art urban office building designed to meet modern workplace demands. It aims to provide a collaborative and sustainable environment for businesses, accommodating approximately 500 employees. The building will feature open-plan offices, conference rooms, coworking spaces, and amenities such as a gym and café. The project is located in a central urban area, which poses unique challenges and opportunities regarding site management, community impact, and regulatory compliance.

        Objectives
        Create a Sustainable Workspace: Incorporate environmentally friendly materials and energy-efficient systems to minimize the building's carbon footprint.
        Enhance Community Engagement: Design the building to serve as a community hub, fostering interactions between tenants and local residents.
        Deliver on Time and Within Budget: Utilize advanced project management techniques to ensure that the project is completed by the scheduled deadline and within the allocated budget.
        Key Project Phases
        Pre-Construction Planning:

        Conduct site assessments to evaluate environmental impacts and regulatory requirements.
        Engage with local authorities and community stakeholders to gather input and address concerns.
        Design Development:

        Collaborate with architects and engineers to develop detailed blueprints that incorporate sustainable practices and innovative design elements.
        Use Building Information Modeling (BIM) technology to visualize the project and identify potential design conflicts early.
        Procurement and Contracting:

        Source materials and services from local suppliers to support the community and reduce transportation emissions.
        Implement transparent bidding processes for subcontractors to ensure competitive pricing and quality.
        Construction Phase:

        Execute the construction work according to the approved designs, closely monitoring adherence to safety standards and quality control measures.
        Utilize project management software to track progress, manage resources, and communicate effectively with all stakeholders.
        Post-Construction Evaluation:

        Conduct a thorough inspection of the completed building to ensure compliance with all design specifications and regulatory requirements.
        Gather feedback from tenants and community members to evaluate the project's success and identify areas for improvement.
        Risk Management
        The project will implement a comprehensive risk management strategy that includes:

        Regulatory Risks: Maintain up-to-date knowledge of local zoning laws and building codes, and actively engage with regulatory agencies.
        Financial Risks: Establish a contingency budget to address unforeseen expenses and regularly review financial projections.
        Construction Risks: Develop a robust safety plan and conduct regular training sessions for construction workers to mitigate on-site accidents.
        Stakeholder Risks: Facilitate regular meetings with stakeholders to ensure that their concerns are addressed throughout the project lifecycle.
        Technology Integration
        The project will leverage advanced technologies to enhance efficiency and communication, including:

        Smart Building Technologies: Implement systems for automated lighting, climate control, and energy monitoring to improve operational efficiency.
        Mobile Project Management Tools: Equip project managers with mobile apps that enable real-time updates and communication with the project team.
        Virtual Reality (VR) for Stakeholder Engagement: Use VR to provide stakeholders with immersive tours of the building design, allowing for feedback before construction begins.
        Sustainability Features
        Green Roof and Vertical Gardens: Enhance the building's aesthetics and provide insulation while improving air quality.
        Rainwater Harvesting Systems: Collect and utilize rainwater for irrigation and non-potable uses within the building.
        Energy-Efficient HVAC Systems: Implement systems designed to reduce energy consumption while maintaining optimal indoor air quality.
        Community Impact
        The project aims to create a positive impact on the local community by:

        Providing job opportunities during and after construction.
        Offering community spaces that can be used for events and gatherings.
        Supporting local businesses through procurement strategies that prioritize local suppliers.

    """

    refined_query = refine_query_for_rag_system(query)

    retrieval_chain = create_retrieval_chain(docsearch)
    answer_with_knowledge = retrieval_chain.invoke({"input": refined_query})

    # Save the answer and context
    save_output_to_file(output_folder, answer_with_knowledge['answer'], answer_with_knowledge['context'])

if __name__ == "__main__":
    main()
