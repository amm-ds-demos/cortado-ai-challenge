import json
from typing import Dict, Any, List
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval

from agent_backend import CortadoAgent
from config import get_config


class AgentEvaluator:
    def __init__(
        self,
    ):
        """
        Initialize the AgentEvaluator with the provided configuration and agent backend.

        This method sets up the evaluator by loading the configuration settings from the config module
        and initializing the agent backend using the CortadoAgent class. The configuration includes
        various settings for chunking, vector tools, JSON tools, memory, agent behavior, and evaluation
        metrics.
        """
        self.config = get_config()
        self.agent_backend = CortadoAgent()

    def _create_geval_metric(self, metric_info: Dict[str, Any]) -> GEval:
        """
        Create a GEval metric based on the provided metric information.

        An arbitrary number of GEval metrics can be defined through the configuration files,
        along its name and criteria.

        Args:
            metric_info (Dict[str, Any]): A dictionary containing information about the metric, such as
                                          its name, criteria, and evaluation parameters.

        Returns:
            GEval: An instance of the GEval metric initialized with the specified parameters.
        """
        return GEval(
            model=self.config["evaluation"]["evaluation_model"],
            name=metric_info["name"],
            criteria=metric_info["criteria"],
            evaluation_params=[
                LLMTestCaseParams[param] for param in metric_info["evaluation_params"]
            ],
            threshold=self.config["evaluation"]["correctness_threshold"],
        )

    def _evaluate_answers(
        self, questions: List[str], expected_answers: List[str], responses: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate the agent's responses to a set of questions using relevancy and custom GEval metrics.

        Args:
            questions (List[str]): A list of questions posed to the agent.
            expected_answers (List[str]): A list of expected answers corresponding to the questions.
            responses (List[str]): A list of actual responses generated by the agent.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the evaluation results for each question,
                                  including the question, expected answer, actual response, and metric scores.
        """
        evaluations = []
        for i, question in enumerate(questions):
            test_case = LLMTestCase(
                input=question,
                actual_output=responses[i],
                expected_output=expected_answers[i],
                context=[],
            )

            answer_relevancy_metric = AnswerRelevancyMetric(
                threshold=self.config["evaluation"]["relevancy_threshold"],
                model=self.config["evaluation"]["evaluation_model"],
            )
            custom_metrics = [
                self._create_geval_metric(metric)
                for metric in self.config["evaluation"]["geval_metrics"]
            ]

            assert_test(test_case, [answer_relevancy_metric] + custom_metrics)

            evaluation = {
                "question": question,
                "expected_answer": expected_answers[i],
                "response": responses[i],
                "metrics": {
                    "relevancy": answer_relevancy_metric.score,
                    **{metric.name: metric.score for metric in custom_metrics},
                },
            }
            evaluations.append(evaluation)

        return evaluations

    def _serialize_tool_output(self, output: Any) -> Dict[str, Any]:
        """
        Serialize the output of a tool used by the agent.

        Args:
            output (Any): The tool output to serialize, which may include content, tool name, raw input,
                          raw output, error status, and source nodes.

        Returns:
            Dict[str, Any]: A dictionary containing the serialized tool output, including content, tool name,
                            raw input, raw output, error status, and source nodes.
        """
        return {
            "content": output.content,
            "tool_name": output.tool_name,
            "raw_input": output.raw_input,
            "raw_output": (
                output.raw_output
                if isinstance(output.raw_output, str)
                else (
                    output.raw_output.response
                    if hasattr(output.raw_output, "response")
                    else str(output.raw_output)
                )
            ),
            "is_error": output.is_error,
            "source_nodes": (
                [
                    {"id": node.node.id_, "text": node.node.text}
                    for node in output.raw_output.source_nodes
                ]
                if hasattr(output.raw_output, "source_nodes")
                else []
            ),
        }

    def _serialize_agent_response(self, response: Any) -> Dict[str, Any]:
        """
        Serialize the response generated by the agent.

        Args:
            response (Any): The agent response to serialize, which includes the response content and sources.

        Returns:
            Dict[str, Any]: A dictionary containing the serialized agent response, including the response content
                            and serialized sources.
        """
        return {
            "response": response.response,
            "sources": [
                self._serialize_tool_output(source) for source in response.sources
            ],
        }

    def _process_agent_responses(self, responses: List[Any]) -> Dict[str, Any]:
        """
        Process and serialize a list of agent responses for further analysis or storage.

        Args:
            responses (List[Any]): A list of agent responses to process and serialize.

        Returns:
            Dict[str, Any]: A dictionary containing the processed and serialized agent responses.
        """
        data = [self._serialize_agent_response(response) for response in responses]
        return {"agent_responses": data}

    def _save_to_json(self, data: Dict[str, Any], filename: str):
        """
        Save the provided data to a JSON file.

        Args:
            data (Dict[str, Any]): The data to save to the JSON file.
            filename (str): The name of the file to save the data to.
        """
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def run_evaluation(self):
        """
        Run the evaluation process by loading questions and expected answers, generating agent responses,
        evaluating the responses, and saving the results to JSON files.

        This method performs the following steps:
        1. Load questions and expected answers from the specified JSON file.
        2. Generate responses from the agent for each question.
        3. Evaluate the responses using relevancy and custom GEval metrics.
        4. Process and serialize the agent responses.
        5. Save the processed responses and evaluation results to JSON files.
        """
        with open(self.config["evaluation"]["questions_file"], "r") as f:
            questions_data = json.load(f)

        questions = questions_data["questions"]
        expected_answers = questions_data["expected_answers"]

        responses = []
        for q in questions:
            try:
                response = self.agent_backend.process_question(q)
                responses.append(response)
            except Exception as e:
                responses.append(f"Error processing question '{q}': {str(e)}")

        evaluations = self._evaluate_answers(
            questions,
            expected_answers,
            [i.response for i in responses if hasattr(i, "response")],
        )
        processed_responses = self._process_agent_responses(responses)
        self._save_to_json(
            processed_responses,
            self.config["evaluation"]["processed_agent_responses_file"],
        )

        with open(self.config["evaluation"]["evaluations_output_file"], "w") as f:
            json.dump(evaluations, f, indent=4)


if __name__ == "__main__":
    evaluator = AgentEvaluator()
    evaluator.run_evaluation()
