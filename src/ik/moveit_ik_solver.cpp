/*
 * Copyright 2019 Southwest Research Institute
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <reach_ros/ik/moveit_ik_solver.h>
#include <reach_ros/utils.h>

#include <moveit/common_planning_interface_objects/common_objects.h>
#include <moveit/planning_scene/planning_scene.h>
#include <reach/plugin_utils.h>
#include <reach/utils.h>
#include <yaml-cpp/yaml.h>
#include <bio_ik/bio_ik.h>
#include <relaxed_ik/relaxed_ik_kinematics_query_options.hpp>

namespace
{
template <typename T>
T clamp(const T& val, const T& low, const T& high)
{
  return std::max(low, std::min(val, high));
}

}  // namespace

namespace reach_ros
{
namespace ik
{
using namespace std::placeholders;

std::string MoveItIKSolver::COLLISION_OBJECT_NAME = "reach_object";

MoveItIKSolver::MoveItIKSolver(moveit::core::RobotModelConstPtr model, const std::string& planning_group,
                               double dist_threshold)
  : model_(model), jmg_(model_->getJointModelGroup(planning_group)), distance_threshold_(dist_threshold)
{
  if (!jmg_)
    throw std::runtime_error("Failed to initialize joint model group for planning group '" + planning_group + "'");
  if (!jmg_->getSolverInstance())
    throw std::runtime_error("No kinematics solver instantiated for planning group '" + planning_group +
                             "'. Check that the 'kinematics.yaml' file was loaded as a parameter");

  scene_.reset(new planning_scene::PlanningScene(model_));

  scene_pub_ =
      reach_ros::utils::getNodeInstance()->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 1);
  hole_pub_ = reach_ros::utils::getNodeInstance()->create_publisher<visualization_msgs::msg::Marker>("hole", 1);
  moveit_msgs::msg::PlanningScene scene_msg;
  scene_->getPlanningSceneMsg(scene_msg);
  scene_pub_->publish(scene_msg);
  auto node = reach_ros::utils::getNodeInstance();
  std::string solver = node->get_parameter("robot_description_kinematics." + planning_group + ".kinematics_solver").as_string();
  solver_name_ = solver.substr(0, solver.find("/"));
  use_rcm_ = node->get_parameter_or("reach_ros.use_rcm", false);
  use_line_goal_ = node->get_parameter_or("reach_ros.use_line_goal", false);
  use_line_alignment_ = node->get_parameter_or("reach_ros.use_line_alignment", false);
  use_depth_ = node->get_parameter_or("reach_ros.use_depth", false);
  use_depth2_ = node->get_parameter_or("reach_ros.use_depth2", false);
  use_collision_distance_ = node->get_parameter_or("reach_ros.use_collision_distance", false);
  use_collision_distance2_ = node->get_parameter_or("reach_ros.use_collision_distance2", false);
  use_empty_cost_fn_ = node->get_parameter_or("reach_ros.empty_cost_fn", false);
  scan_goal_ = node->get_parameter_or("reach_ros.scan_goal", false);
  scan_with_offset_ = node->get_parameter_or("reach_ros.scan_with_offset", false);
  scan_swamp_ = node->get_parameter_or("reach_ros.scan_swamp", false);
  use_rcm2_ = node->get_parameter_or("reach_ros.use_rcm2", false);
  use_rcm3_ = node->get_parameter_or("reach_ros.use_rcm3", false);
}

std::vector<std::vector<double>> MoveItIKSolver::solveIK(const Eigen::Isometry3d& target_,
                                                         const std::map<std::string, double>& seed) const
{
  Eigen::Isometry3d target(target_);
  moveit::core::RobotState state(model_);

  const std::vector<std::string>& joint_names = jmg_->getActiveJointModelNames();

  std::vector<double> seed_subset = reach::extractSubset(seed, joint_names);
  state.setJointGroupPositions(jmg_, seed_subset);
  state.update();

  solution_callback_count_ = 0;

  auto options = std::make_shared<kinematics::KinematicsQueryOptions>();
  auto bio_ik_options = std::make_shared<bio_ik::BioIKKinematicsQueryOptions>();
  auto relaxed_ik_options = std::make_shared<relaxed_ik::RelaxedIKKinematicsQueryOptions>();
  if (solver_name_ == "bio_ik")
  {
    tf2::Vector3 position{target.translation().x(), target.translation().y(), target.translation().z()};
    Eigen::Quaterniond q(target.rotation());
    tf2::Quaternion orientation(q.x(), q.y(), q.z(), q.w());

    if (scan_goal_) {
      bio_ik_options->replace = true;
      bio_ik_options->goals.emplace_back(new bio_ik::ScanGoal("end_effector", position, orientation, 1.0));
    }
    if (scan_swamp_) {
      bio_ik_options->replace = true;
      bio_ik_options->goals.emplace_back(new bio_ik::ScanSwampGoal("end_effector", position, orientation, 1.0));
    }
    if (scan_with_offset_) {
      Eigen::Vector3d offset{0.0, 0.0, 0.01};
      target.translate(-offset);
    }

    if (hole_position_ && use_rcm_)
    {
      bio_ik_options->goals.emplace_back(new bio_ik::RCMGoal(hole_position_.value(), 1.0));
    }
    if (hole_position_ && use_rcm3_)
    {
      bio_ik_options->goals.emplace_back(new bio_ik::RCMGoal3("endo_first_link", hole_position_.value(), 1.0));
    }
    if (hole_position_ && hole_axis_ && use_line_goal_) {
      bio_ik_options->goals.emplace_back(new bio_ik::LineGoal("endo_first_link", hole_position_.value(),
                                                              hole_axis_.value(), 1.0));
    }
    if (hole_axis_ && use_line_alignment_) {
      bio_ik_options->goals.emplace_back(new bio_ik::DirectionGoal("endo_first_link", tf2::Vector3(0, 1, 0), hole_axis_.value(), 1.0));
    }
    if (use_depth_)
    {
      kinematics::KinematicsBase::IKCostFn depth_fn =
          [this](const geometry_msgs::msg::Pose&, const moveit::core::RobotState& robot_state,
                 const moveit::core::JointModelGroup*, const std::vector<double>&) {
            collision_detection::CollisionRequest req;
            req.contacts = true;
            collision_detection::CollisionResult res;
            scene_->checkCollision(req, res, robot_state);
            double penetration_depth = 0;
            for (const auto& [link_names, contacts] : res.contacts)
            {
              for (const auto& contact : contacts)
              {
                penetration_depth += contact.depth;
              }
            }
            double cost = std::pow(penetration_depth / 0.05, 2);
            return cost;
          };
      const geometry_msgs::msg::Pose p;
      bio_ik_options->goals.emplace_back(new bio_ik::IKCostFnGoal(p, depth_fn, model_));
    }
    if (use_depth2_)
    {
      kinematics::KinematicsBase::IKCostFn depth_fn =
          [this](const geometry_msgs::msg::Pose&, const moveit::core::RobotState& robot_state,
                 const moveit::core::JointModelGroup*, const std::vector<double>&) {
            collision_detection::AllowedCollisionMatrix acm;
            acm.setDefaultEntry("base_link_inertia", true);
            acm.setDefaultEntry("shoulder_link", true);
            acm.setDefaultEntry("upper_arm_link", true);
            acm.setDefaultEntry("forearm_link", true);
            acm.setDefaultEntry("wrist_1_link", true);
            acm.setDefaultEntry("wrist_2_link", true);
            acm.setDefaultEntry("wrist_3_link", true);
            acm.setEntry("endo_box", "endo_first_link", true);
            acm.setEntry("endo_first_link", "endo_second_link", true);
            acm.setEntry("endo_second_link", "endo_third_link", true);
            collision_detection::CollisionRequest req;
            req.contacts = true;
            collision_detection::CollisionResult res;
            scene_->checkCollision(req, res, robot_state, acm);
            double penetration_depth = 0;
            for (const auto& [link_names, contacts] : res.contacts)
            {
              for (const auto& contact : contacts)
              {
                penetration_depth += contact.depth;
              }
            }
            double cost = std::pow(penetration_depth / 0.05, 2);
            return cost;
          };
      const geometry_msgs::msg::Pose p;
      bio_ik_options->goals.emplace_back(new bio_ik::IKCostFnGoal(p, depth_fn, model_));
    }
    if (use_collision_distance_)
    {
      kinematics::KinematicsBase::IKCostFn collision_distance_fn =
          [this](const geometry_msgs::msg::Pose&, const moveit::core::RobotState& robot_state,
                 const moveit::core::JointModelGroup*, const std::vector<double>&) {
            collision_detection::AllowedCollisionMatrix acm;
            acm.setDefaultEntry("end_effector", true);
            double distance = scene_->distanceToCollision(robot_state, acm);
            double penalty_cutoff = 0.01;
            // distance cost is 1 if distance == penalty_cutoff
            double distance_cost = std::pow(2 * penalty_cutoff / (distance + penalty_cutoff), 2);
            return distance_cost;
          };
      const geometry_msgs::msg::Pose p;
      bio_ik_options->goals.emplace_back(new bio_ik::IKCostFnGoal(p, collision_distance_fn, model_));
    }
    if (use_collision_distance2_)
    {
      kinematics::KinematicsBase::IKCostFn collision_distance_fn =
          [this](const geometry_msgs::msg::Pose&, const moveit::core::RobotState& robot_state,
                 const moveit::core::JointModelGroup*, const std::vector<double>&) {
            collision_detection::AllowedCollisionMatrix acm;
            acm.setDefaultEntry("end_effector", true);
            acm.setDefaultEntry("base_link_inertia", true);
            acm.setDefaultEntry("shoulder_link", true);
            acm.setDefaultEntry("upper_arm_link", true);
            acm.setDefaultEntry("forearm_link", true);
            acm.setDefaultEntry("wrist_1_link", true);
            acm.setDefaultEntry("wrist_2_link", true);
            acm.setDefaultEntry("wrist_3_link", true);
            double distance = scene_->distanceToCollision(robot_state, acm);
            double penalty_cutoff = 0.01;
            // distance cost is 1 if distance == penalty_cutoff
            double distance_cost = std::pow(2 * penalty_cutoff / (distance + penalty_cutoff), 2);
            return distance_cost;
          };
      const geometry_msgs::msg::Pose p;
      bio_ik_options->goals.emplace_back(new bio_ik::IKCostFnGoal(p, collision_distance_fn, model_));
    }
    if (use_empty_cost_fn_)
    {
      kinematics::KinematicsBase::IKCostFn empty_cost_fn =
          [](const geometry_msgs::msg::Pose&, const moveit::core::RobotState&,
                 const moveit::core::JointModelGroup*, const std::vector<double>&) {
            return 0.0;
          };
      const geometry_msgs::msg::Pose p;
      bio_ik_options->goals.emplace_back(new bio_ik::IKCostFnGoal(p, empty_cost_fn, model_));
    }

    options = bio_ik_options;
  }
  else if (solver_name_ == "relaxed_ik") {
    relaxed_ik_options->objectives_.emplace_back(std::make_shared<relaxed_ik::MatchEEPosGoals>(), 1.0);
    relaxed_ik_options->objectives_.emplace_back(std::make_shared<relaxed_ik::MatchEEQuatGoals>(), 1.0);
    if (hole_position_ && use_rcm_) {
      relaxed_ik_options->objectives_.emplace_back(
          std::make_shared<relaxed_ik::RCMGoal>(Eigen::Vector3d(hole_position_->x(), hole_position_->y(), hole_position_->z())),
             1.0);
    }
    if (hole_position_ && use_rcm3_) {
      relaxed_ik_options->objectives_.emplace_back(
          std::make_shared<relaxed_ik::RCMGoal3>("endo_first_link", Eigen::Vector3d(hole_position_->x(), hole_position_->y(), hole_position_->z())),
             1.0);
    }
    if (hole_position_ && hole_axis_ && use_line_goal_) {
      relaxed_ik_options->objectives_.emplace_back(
          std::make_shared<relaxed_ik::LineGoal>(Eigen::Vector3d(hole_position_->x(), hole_position_->y(),
                                                              hole_position_->z()),
                                                 Eigen::Vector3d(hole_axis_->x(), hole_axis_->y(), hole_axis_->z()),
                                                 "endo_first_link"),
          1.0);
    }
    if (hole_axis_ && use_line_alignment_) {
      relaxed_ik_options->objectives_.emplace_back(
        std::make_shared<relaxed_ik::AlignmentGoal>(Eigen::Vector3d(
                                                          hole_axis_->x(),
                                                          hole_axis_->y(),
                                                          hole_axis_->z()),
                                               "endo_first_link"),
        1.0);
    }
    if (hole_position_ && use_rcm2_)
    {
      relaxed_ik_options->objectives_.emplace_back(
          std::make_shared<relaxed_ik::RCMGoal2>(model_, Eigen::Vector3d(hole_position_->x(), hole_position_->y(), hole_position_->z())),
             1.0);
    }
    if (use_depth_) {
      relaxed_ik_options->objectives_.emplace_back(std::make_shared<relaxed_ik::EnvCollisionDepth>(scene_), 1.0);
    }
    if (use_depth2_) {
      relaxed_ik_options->objectives_.emplace_back(std::make_shared<relaxed_ik::EnvCollisionDepth2>(scene_), 1.0);
    }
    if (use_collision_distance_) {
      relaxed_ik_options->objectives_.emplace_back(std::make_shared<relaxed_ik::EnvCollisionDistance>(scene_), 1.0);
    }
    if (use_collision_distance2_) {
      relaxed_ik_options->objectives_.emplace_back(std::make_shared<relaxed_ik::EnvCollisionDistance2>(scene_), 1.0);
    }
    if (use_empty_cost_fn_) {
      kinematics::KinematicsBase::IKCostFn empty_cost_fn =
          [](const geometry_msgs::msg::Pose&, const moveit::core::RobotState&,
                 const moveit::core::JointModelGroup*, const std::vector<double>&) {
            return 0.0;
          };
      const geometry_msgs::msg::Pose p;
	    relaxed_ik_options->objectives_.emplace_back(std::make_shared<relaxed_ik::IKCostFnGoal>(p, empty_cost_fn), 1.0);
    }
    options = relaxed_ik_options;
  }

  if (state.setFromIK(jmg_, target, 0.0,
                      std::bind(&MoveItIKSolver::isIKSolutionValid, this, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3), *options))
  {
    std::vector<double> solution;
    state.copyJointGroupPositions(jmg_, solution);

    return { solution };
  }

  return {};
}

bool MoveItIKSolver::isIKSolutionValid(moveit::core::RobotState* state, const moveit::core::JointModelGroup* jmg,
                                       const double* ik_solution) const
{
  solution_callback_count_++;
  state->setJointGroupPositions(jmg, ik_solution);
  state->update();

  const bool colliding = scene_->isStateColliding(*state, jmg->getName(), false);
  const bool too_close =
      (scene_->distanceToCollision(*state, scene_->getAllowedCollisionMatrix()) < distance_threshold_);

  return (!colliding && !too_close);
}

std::vector<std::string> MoveItIKSolver::getJointNames() const
{
  return jmg_->getActiveJointModelNames();
}

void MoveItIKSolver::setHoleAxis(const tf2::Vector3 hole_axis) {
  hole_axis_ = hole_axis;
}

void MoveItIKSolver::setHolePosition(const tf2::Vector3 hole_position)
{
  hole_position_ = hole_position;
  visualization_msgs::msg::Marker m;
  m.header.frame_id = "base_link";
  m.ns = "hole";
  m.type = visualization_msgs::msg::Marker::SPHERE;
  m.color.r = 1.0;
  m.color.a = 1.0;
  m.pose.position.x = hole_position.x();
  m.pose.position.y = hole_position.y();
  m.pose.position.z = hole_position.z();
  m.scale.x = 0.01;
  m.scale.y = 0.01;
  m.scale.z = 0.01;
  hole_pub_->publish(m);
}

void MoveItIKSolver::addCollisionMesh(const std::string& collision_mesh_filename,
                                      const std::string& collision_mesh_frame)
{
  // Add the collision object to the planning scene
  moveit_msgs::msg::CollisionObject obj =
      utils::createCollisionObject(collision_mesh_filename, collision_mesh_frame, COLLISION_OBJECT_NAME);
  if (!scene_->processCollisionObjectMsg(obj))
    throw std::runtime_error("Failed to add collision mesh to planning scene");

  moveit_msgs::msg::PlanningScene scene_msg;
  scene_->getPlanningSceneMsg(scene_msg);
  scene_pub_->publish(scene_msg);
}

void MoveItIKSolver::setTouchLinks(const std::vector<std::string>& touch_links)
{
  scene_->getAllowedCollisionMatrixNonConst().setEntry(COLLISION_OBJECT_NAME, touch_links, true);
}

std::string MoveItIKSolver::getKinematicBaseFrame() const
{
  return jmg_->getSolverInstance()->getBaseFrame();
}

reach::IKSolver::ConstPtr MoveItIKSolverFactory::create(const YAML::Node& config) const
{
  auto planning_group = reach::get<std::string>(config, "planning_group");
  auto dist_threshold = reach::get<double>(config, "distance_threshold");

  moveit::core::RobotModelConstPtr model =
      moveit::planning_interface::getSharedRobotModel(reach_ros::utils::getNodeInstance(), "robot_description");
  if (!model)
    throw std::runtime_error("Failed to initialize robot model pointer");

  auto ik_solver = std::make_shared<MoveItIKSolver>(model, planning_group, dist_threshold);
  if (config["hole_position"].IsDefined())
  {
    auto hole_position = reach::get<std::vector<double>>(config, "hole_position");
    if (!hole_position.empty())
    {
      ik_solver->setHolePosition(tf2::Vector3(hole_position[0], hole_position[1], hole_position[2]));
    }
  }
  if (config["hole_axis"].IsDefined())
  {
    auto hole_axis = reach::get<std::vector<double>>(config, "hole_axis");
    if (!hole_axis.empty())
    {
      ik_solver->setHoleAxis(tf2::Vector3(hole_axis[0], hole_axis[1], hole_axis[2]));
    }
  }

  // Optionally add a collision mesh
  const std::string collision_mesh_filename_key = "collision_mesh_filename";
  const std::string collision_mesh_frame_key = "collision_mesh_frame";
  if (config[collision_mesh_filename_key])
  {
    auto collision_mesh_filename = reach::get<std::string>(config, collision_mesh_filename_key);
    std::string collision_mesh_frame = config[collision_mesh_frame_key] ?
                                           reach::get<std::string>(config, collision_mesh_frame_key) :
                                           ik_solver->getKinematicBaseFrame();

    ik_solver->addCollisionMesh(collision_mesh_filename, collision_mesh_frame);
  }

  // Optionally add touch links
  const std::string touch_links_key = "touch_links";
  if (config[touch_links_key])
  {
    auto touch_links = reach::get<std::vector<std::string>>(config, touch_links_key);
    ik_solver->setTouchLinks(touch_links);
  }

  return ik_solver;
}

DiscretizedMoveItIKSolver::DiscretizedMoveItIKSolver(moveit::core::RobotModelConstPtr model,
                                                     const std::string& planning_group, double dist_threshold,
                                                     double dt)
  : MoveItIKSolver(model, planning_group, dist_threshold), dt_(dt)
{
}

std::vector<std::vector<double>> DiscretizedMoveItIKSolver::solveIK(const Eigen::Isometry3d& target,
                                                                    const std::map<std::string, double>& seed) const
{
  // Calculate the number of discretizations necessary to achieve discretization angle
  const static int n_discretizations = int((2.0 * M_PI) / dt_);

  std::vector<std::vector<double>> solutions;
  solutions.reserve(n_discretizations);

  for (int i = 0; i < n_discretizations; ++i)
  {
    Eigen::Isometry3d discretized_target(target * Eigen::AngleAxisd(double(i) * dt_, Eigen::Vector3d::UnitZ()));
    std::vector<std::vector<double>> tmp_sols = MoveItIKSolver::solveIK(discretized_target, seed);

    if (!tmp_sols.empty())
      solutions.push_back(tmp_sols.front());
  }

  return solutions;
}

reach::IKSolver::ConstPtr DiscretizedMoveItIKSolverFactory::create(const YAML::Node& config) const
{
  auto planning_group = reach::get<std::string>(config, "planning_group");
  auto dist_threshold = reach::get<double>(config, "distance_threshold");

  moveit::core::RobotModelConstPtr model =
      moveit::planning_interface::getSharedRobotModel(reach_ros::utils::getNodeInstance(), "robot_description");
  if (!model)
    throw std::runtime_error("Failed to initialize robot model pointer");

  auto dt = std::abs(reach::get<double>(config, "discretization_angle"));
  double clamped_dt = clamp<double>(dt, 0.0, M_PI);
  if (std::abs(dt - clamped_dt) > 1.0e-6)
  {
    std::cout << "Clamping discretization angle between 0 and pi; new value is " << clamped_dt;
  }
  dt = clamped_dt;

  auto ik_solver = std::make_shared<DiscretizedMoveItIKSolver>(model, planning_group, dist_threshold, dt);

  // Optionally add a collision mesh
  const std::string collision_mesh_filename_key = "collision_mesh_filename";
  const std::string collision_mesh_frame_key = "collision_mesh_frame";
  if (config[collision_mesh_filename_key])
  {
    auto collision_mesh_filename = reach::get<std::string>(config, collision_mesh_filename_key);
    std::string collision_mesh_frame = config[collision_mesh_frame_key] ?
                                           reach::get<std::string>(config, collision_mesh_frame_key) :
                                           ik_solver->getKinematicBaseFrame();

    ik_solver->addCollisionMesh(collision_mesh_filename, collision_mesh_frame);
  }

  const std::string touch_links_key = "touch_links";
  if (config[touch_links_key])
  {
    auto touch_links = reach::get<std::vector<std::string>>(config, touch_links_key);
    ik_solver->setTouchLinks(touch_links);
  }

  return ik_solver;
}

}  // namespace ik
}  // namespace reach_ros
