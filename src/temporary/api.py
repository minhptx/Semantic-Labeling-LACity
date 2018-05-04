from pathlib import Path

from semantic_labeling.evaluation import eval_sources
from semantic_labeling.typer import SemanticTyper
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.utilities.serializable import serializeJSON

if __name__ == "__main__":
    dataset = "lacity"

    # sms_model_list = [sm.get_semantic_model() for sm in get_karma_models(dataset)]
    sms_model_list = get_semantic_models(dataset)
    print(sms_model_list)
    # sms_model_list: List[SemanticModel] = []
    # datadir = Path(config.datasets.lacity.as_path())
    # for file in (datadir / "models_semantic_types").iterdir():
    #     table = DataTable.load_from_file(datadir / "sources" / f"{file.stem}.csv")
    #     sms_model_list.append(R2RML.load_from_file(file).apply_cmds(table))

    typer = SemanticTyper.get_instance(dataset, sms_model_list[1:])

    typer.semantic_labeling(sms_model_list[1:], sms_model_list[:1], 4, eval_train=True)

    exp_dir = Path("output")
    eval_sources(sms_model_list, exp_dir / f"eval.train.csv")

    with open("output/debug.txt", "w") as f:
        for sm in sms_model_list:
            f.write("Source: %s\n" % sm.id)
            for attr in sm.attrs:
                f.write("    + attr: %s: [%s]\n" % (attr.label, ", ".join(
                    f"{st.domain}--{st.type}: {st.confidence_score:.3f}" for st in attr.semantic_types)))

