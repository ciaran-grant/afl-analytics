from afl_analytics.arpadl.pyafl import convert_to_actions
from afl_analytics.arpadl.schema import ARPADLSchema
from AFLPy.AFLData_Client import load_data
 
def test_pyafl_convert_to_actions():
     
    chains = load_data('AFL_API_Match_Chains', ID = "AFL_2022_F4_Geelong_Sydney")
    actions = convert_to_actions(chains)
    
    assert (ARPADLSchema.validate(actions) == actions).all().all()
    
def test_pyafl_convert_to_actions_2021():
     
    chains = load_data('AFL_API_Match_Chains', ID = "AFL_2021")
    actions = convert_to_actions(chains)
    
    assert (ARPADLSchema.validate(actions) == actions).all().all()
    
def test_pyafl_convert_to_actions_2022():
     
    chains = load_data('AFL_API_Match_Chains', ID = "AFL_2022")
    actions = convert_to_actions(chains)
    
    assert (ARPADLSchema.validate(actions) == actions).all().all()
    
def test_pyafl_convert_to_actions_2023():
     
    chains = load_data('AFL_API_Match_Chains', ID = "AFL_2023")
    actions = convert_to_actions(chains)
    
    assert (ARPADLSchema.validate(actions) == actions).all().all()
    
def test_pyafl_convert_to_actions_2024():
     
    chains = load_data('AFL_API_Match_Chains', ID = "AFL_2024")
    actions = convert_to_actions(chains)
    
    assert (ARPADLSchema.validate(actions) == actions).all().all()