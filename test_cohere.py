from utils.get_embedd_model import embedding_model
from sklearn.metrics.pairwise import cosine_similarity


def test_cohere():

    print("Initializing Cohere multilingual embedder...")

    embedder = embedding_model()

    if not embedder:
        print("❌ Failed to initialize embedder. Check COHERE_API_KEY.")
        return


    test_inputs = [

        "How to apply for PAN card in India",        # English

        "PAN card apply kaise kare",                 # Hinglish

        "पॅन कार्ड कसे मिळवावे",                    # Marathi

        "भारत में पैन कार्ड कैसे बनाएं"              # Hindi

    ]


    print("\nGenerating multilingual embeddings...\n")

    try:

        vectors = embedder.embed_documents(test_inputs)

        if not vectors:
            print("❌ No embeddings returned")
            return


        dim = len(vectors[0])

        print("✅ SUCCESS!")
        print(f"Embedding dimension: {dim}\n")


        print("=========== SENTENCE + EMBEDDING PREVIEW ===========\n")

        for sentence, vector in zip(test_inputs, vectors):

            print(f"Sentence:\n{sentence}\n")

            print(f"Vector dimension: {len(vector)}")

            print("First 10 values of embedding:")

            print(vector[:10])

            print("-" * 70)


        print("\n=========== SIMILARITY CHECK ===========\n")

        for i in range(len(test_inputs)):
            for j in range(i+1, len(test_inputs)):

                score = cosine_similarity(

                    [vectors[i]],
                    [vectors[j]]

                )[0][0]

                print(f"Similarity between:")
                print(f"'{test_inputs[i]}'")
                print("AND")
                print(f"'{test_inputs[j]}'")

                print(f"Score: {score:.4f}")

                print("-" * 50)


        if dim == 1024:

            print("\n🎯 Embedding dimension matches pgvector schema (1024)")

        else:

            print(f"\n⚠️ Update PostgreSQL schema to vector({dim})")


    except Exception as e:

        print(f"❌ Cohere API call failed: {e}")


if __name__ == "__main__":

    test_cohere()