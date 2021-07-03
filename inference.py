import pandas as pd
from torch.nn import MSELoss
from easydict import EasyDict as edict
from readability_transformers import ReadabilityTransformer
from readability_transformers.readers import PairwiseDataReader, PredictionDataReader
from readability_transformers.dataset import CommonLitDataset
from readability_transformers.losses import WeightedRankingMSELoss

def inference_on_dataset():
    inf_options = {
        "model_path": "checkpoints/new_1",
        "device": "cuda:1"
    }
    inf_options = edict(inf_options)

    model = ReadabilityTransformer(
        model_path=inf_options.model_path,
        device=inf_options.device
    )
    
    dataset = CommonLitDataset("test", cache=True)
    test_df = dataset.data
    parameters = dataset.load_parameters(inf_options["model_path"])

    lingfeat_subgroups = parameters["lingfeat_subgroups"]
    lingfeat_features = parameters["lingfeat_features"]
    # different from lingfeat_maxes.keys() because of parameters["blacklist"]
    lingfeat_maxes = parameters["lingfeat_maxes"]
    lingfeat_mins = parameters["lingfeat_mins"]

    test_df = dataset.apply_lingfeat_features(
        [test_df], 
        subgroups=lingfeat_subgroups,
        features=lingfeat_features,
        feature_maxes=lingfeat_maxes,
        feature_mins=lingfeat_mins
    )[0]

    test_df_feats = [feat for feat in test_df.columns.values if feat.startswith("feature_")]
    print(len(test_df_feats))
    print(len(lingfeat_features))

    test_embed = dataset.apply_st_embeddings(
        [test_df],
        st_model=model.st_model,
        batch_size=7
    )[0]

    inference_reader = PredictionDataReader(test_df, test_embed, features=lingfeat_features, no_targets=True)
    
    dataloader = inference_reader.get_dataloader(batch_size=len(test_df))

    for batch in dataloader:
        inputs = batch["inputs"].to(inf_options.device)
        predictions = model.rp_model(inputs)
        print(predictions)


def inference():
    options = {
        "model_checkpoint": "checkpoints/save",
        "device": "cuda:1"
    }
    options = edict(options)

    model = ReadabilityTransformer(
        model_path=options.model_path,
        device=options.device
    )

    texts = [
        "BERT out-of-the-box is not the best option for this task, as the run-time in your setup scales with the number of sentences in your corpus. I.e., if you have 10,000 sentences/articles in your corpus, you need to classify 10k pairs with BERT, which is rather slow."
    ]
    predictions = model(texts, batch_size=1)
    print(predictions)


if __name__ == "__main__":
    inference_on_dataset()