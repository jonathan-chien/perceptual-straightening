class ParamsDict:
    """
    Helper class for more readable access of parameters. Objects of this class
    wrap a dictionary; an object's internal dictionary can then be accessed with
    dot indexing instead of traditional string-based key indexing. TODO: Check
    if following statement is true: Note that this means that access to
    attributes is not controlled through any getter routine, so excercise
    caution if ever extending this class.  
    """
    def __init__(self, params_dict):
        """
        params_dict : dict whose key value pairs are parameters for a class.
        """
        assert isinstance(params_dict, dict), "`params_dict` must be a dict."
        self.__dict__.update(params_dict)


class Params:
    """
    """
    def __init__(
            self, 
            default_params=None, 
            params=None, 
            kwargs_lock=True, 
            **kwargs
        ):
        self.default_params = default_params or {}
        self.passed_in_params = {} # Track params passed in via kwargs
        self.kwargs_params = {} # Track params passed in via dict 
        self.params = {} # The final dictionary of parameters.

        self.kwargs_lock = kwargs_lock
        self.updated_from_dict = False
        self.updated_from_kwargs = False

        self._update_from_dict(params)
        self._update_from_kwargs(**kwargs)

    def _update_from_dict(self, params=None):
        """
        """
        if self.updated_from_dict: 
            raise Exception("This method may only be called once.")
        params = params or {}
        self.passed_in_params = {**self.passed_in_params, **params}
        self.params = {**self.default_params, **self.passed_in_params}
        self.updated_from_dict = True

    def _update_from_kwargs(self, **kwargs):
        """
        Allows updates to parameters. Can only be called once in order to 
        manage and preserve precedence hierarchy for update sources.

        Arguments
        ---------
        **kwargs : key-word arguments.
        """
        if (
            self.kwargs_lock 
            and not all([value is None for value in kwargs.values()])
        ):
            raise Exception(
                "kwargs with values not None were detected, but `kwargs_lock` "
                "is True. If you wish to pass in parameters with kwargs, set "
                "`self.kwargs_lock` to False."
            )
        
        elif not self.updated_from_dict:
            raise Exception(
                "`_update_from_dict` must be called before `_update_from_kwargs`."
            )
        elif self.updated_from_kwargs:
            raise Exception(
                "This method may only be called once."
            )

        for param_name, param in kwargs.items():
            # If parameter value was set using dict but kwarg also attempted.
            if param_name in self.passed_in_params.keys() and param is not None:
                raise Exception(
                    f"{param_name} was passed in as a key-word argument, but "
                    "it was already specified in the passed-in instance params "
                    "dictionary."
                )
            # Ensure kwarg key matches some key in default param dict (which
            # is included in the merged dict).
            elif param_name not in self.params.keys():
                raise Exception(
                    f"Key for kwarg `{param_name}` does not match any key "
                    "in the default_params dict."
                )
            # Replace default value with kwarg value.
            else:
                setattr(self.params, param_name, param)
                self.updated_from_kwargs = True


def set_params(default_params, params, kwargs_lock=True, **kwargs):
    """
    """
    params_manager = Params(
        default_params, 
        params, 
        kwargs_lock=kwargs_lock, 
        **kwargs
    )
    effective_params = ParamsDict(params_manager.params)   
    return effective_params, params_manager



# class Params(ABC):
#     """
#     Precedence hierarchy: passed-in params > kwargs > default params. If user 
#     attempts to overwrite passed in params with kwargs, an exception will be 
#     raised. If attribute `kwargs_lock` is True, attempting to pass in any kwargs
#     not None will result in an exception.
#     """
#     # This will be extended by inheriting class using dictionary merging. 
#     default_params = {}

#     def __init__(self, params={}, kwargs_lock=True):
#         # """

#         # Arguments
#         # ---------
#         # params      : {Dict | Default
#         # kwargs_lock : {Boolean | Default=True}
#         # """
#         # self._set_params(params)
#         # self.kwargs_lock = kwargs_lock
#         pass
    
#     def _set_params(self, params):
#         """
#         Takes in a dictionary containing passed in parameters for an instance 
#         and merges these with the default class level parameters, giving 
#         priority to the former for any overlaps.

#         Arguments/Operates on
#         ---------------------
#         params              : {dict} Dict containing parameters for an instance
#                               of the inheriting class.
#         self.default_params : {dict | Default={}} Class attribute consisting of
#                               dict with default parameters.
        

#         Returns/New attributes
#         ----------------------
#         self.passed_in_params : {Dict} A copy of passed in params so that
#                                 warning extra warning can be issued if  kwarg
#                                 parameter supersedes passed in parameter. Can
#                                 also be helpful to read at a glance what
#                                 parameters were specified explicitly for a given
#                                 object (as opposed to set from default values).
#         self.params           : {ParamsDict} An instance of the ParamsDict class
#                                 whose attributes store the merged/priority
#                                 checked versions of default and passed in
#                                 parameters.
#         """
#         assert isinstance(self.default_params, dict), (
#             "`Class attribute default_params must be a dict."
#         )
#         assert isinstance(params, dict), (
#             "`params` must be a dict."
#         )

#         # Retain a copy of passed in params (see documentation).
#         self.passed_in_params = params

#         # Merge default params with passed in params, giving priority to the
#         # latter for any overlaps.
#         effective_params = {**self.default_params, **params}
#         self.params = ParamsDict(effective_params)

#     def _update_from_kwarg(self, kwargs_dict):
#         """
#         Method to be called at beginning of every method in inheriting class
#         that accepts kwargs that correspond to relevant parameters.

#         Arguments
#         ---------
#         kwargs_dict : {Dict} This dictionary consists of the kwargs that were 
#                       passed to the method calling this one.
#         """
#         if (
#             self.kwargs_lock 
#             and not all([kwargs_dict[key] is None for key in kwargs_dict])
#         ):
#             raise Exception(
#                 "kwargs with values not None were detected, but `kwargs_lock` "
#                 "is True. If you wish to pass in parameters with kwargs, set "
#                 "`self.kwargs_lock` to False."
#             )

#         for param_name, param in kwargs_dict.items():
#             # If parameter value was specified in passed in dict but setting
#             # parameter with kw arg argument was also attempted: warn user that
#             # the kwarg will be ignored.
#             if (param_name in self.passed_in_params.keys() 
#                 and param is not None
#                 and param != self.params.__dict__[param_name]
#             ):
#                 raise Exception(
#                     f"{param_name} was passed in as a key-word argument, but "
#                     "it was already specified in the passed-in instance params "
#                     "dictionary."
#                 )
#             # Ensure kwarg key matches some key in default param dict (which
#             # is included in the merged dict).
#             elif param_name not in self.params.__dict__.keys():
#                 raise Exception(
#                     f"Key for kwarg `{param_name}` does not match any key "
#                     "in the default_params dict."
#                 )
#             # Replace default value with kwarg value.
#             else:
#                 setattr(self.params, param_name, param)



    # def check_kwarg_and_set_params(self, **kwargs):
    #     """
    #     self.default_params and self.params are instances of the ParamsDict 
    #     class.
    #     """
    #     self._check_for_default_params()
    #     self._check_passed_in_params()

    #     # Merge the default_params attribute dict with the kwargs dict so that 
    #     # the latter takes priority over the latter for any overlaps.
    #     merged_params = {**self.default_params.__dict__, **kwargs}

    #     # Merge the merging of default_params attribute dict and kwargs dict 
    #     # with the passed in params attribute dict so that the latter takes 
    #     # priority for overlaps.
    #     if self.params is None:
    #         # If instance level params dict was not provided, point any 
    #         # references to this params dict to the merging of the 
    #         # default_params and kwargs dicts.
    #         effective_params = merged_params
    #     else:
    #         effective_params = {**merged_params, **self.params.__dict__}

    #     # If parameter value was specified in passed in dict but setting 
    #     # parameter with kw arg argument was also attempted: warn user that the
    #     # kwarg will be ignored.
    #     for param_name, param in kwargs.items():
    #         if param is not None and param != self.params.__dict__[param_name]:
    #             warn(f"{param_name} {self.warning_string}")

    #     # Update params (ParamsDict object).
    #     self.params = ParamsDict(effective_params)

        # for param_name in default_params_list:
        #     if param_name in passed_in_params_list:
        #         # Parameter value was specified in passed in dict but setting
        #         # parameter with kw arg argument also attempted: warn user
        #         # that the kwarg will be ignored.
        #         if param_name in kwargs_list: 
        #             warn(f"{param_name} {self.warning_string}")
        #     elif param_name in kwargs_list:
        #         # If parameter not specified in passed in dict, but was 
        #         # specified via kwarg, this takes precedence over default value.
        #         setattr(self, param_name, kwargs[param_name])


    # def _check_for_default_params(self):
    #     """
    #     Exception handling if default_params is not a dictionary.  
    #     """
    #     assert()
    #     if self.default_params is None:
    #         raise Exception(
    #             "`default_params` dict has value None, indicating it was not "
    #             "set (or not set properly) by the inheriting class."
    #         )
    #     elif not isinstance(self.default_params, dict):
    #         raise Exception(
    #             "`default_params` must be a dict."
    #         )
    
    # def _check_passed_in_params(self):
    #     """
    #     Extra check to ensure that if any instance level parameters were passed
    #     in in params dict that the inheriting class used this dict to 
    #     instantiate an object of the ParamsDict class.
    #     """
    #     if self.params is not None: 
    #         assert (
    #             isinstance(self.params, dict),
    #             "If self.params is not None, it must be a dict."
    #         )







# def parameter_kw_validation(kw_arg_value, param_name, params):
#     if kw_arg_value is not None:
#         assert (
#             param_name in params.keys(), 
#             f"{param_name} was specified neither in the params dict nor as a kwarg."
#         )
#         parameter = params[param_name]
#     else:
#         if 
