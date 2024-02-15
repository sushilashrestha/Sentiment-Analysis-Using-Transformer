test = input("Do You Want to Test the Model Now (y/n): ")
if test.lower() == "y":
        print("Getting Ready for Inference... Enter `q` to exit.")
        model = prepare_and_train.prepared_model.load_state_dict(torch.load(model_path))
        while True:
            sentence = input("Enter your Sentence: ")
            if sentence != "q":
                prepare_and_train.predict(sentence, model, prepare_and_train.tokenize)
            break
