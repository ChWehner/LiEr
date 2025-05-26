from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna_dashboard import run_server
import optuna

storage = JournalStorage(JournalFileBackend("hpo_journal.log"))

study = optuna.load_study(study_name="Countries_S2_interactive", storage=storage)
print("Best params so far:", study.best_params)

run_server(storage)
