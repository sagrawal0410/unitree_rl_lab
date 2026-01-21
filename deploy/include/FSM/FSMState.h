#pragma once

#include <algorithm>
#include <string>
#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

// Helper function to check if a condition involves keyboard directional keys
inline bool condition_uses_keyboard_keys(const std::string& condition) {
    std::string lower_condition = condition;
    std::transform(lower_condition.begin(), lower_condition.end(), lower_condition.begin(), ::tolower);
    return lower_condition.find("up") != std::string::npos ||
           lower_condition.find("down") != std::string::npos ||
           lower_condition.find("left") != std::string::npos ||
           lower_condition.find("right") != std::string::npos;
}

// Helper function to check keyboard for directional keys matching the condition
// This handles simple conditions like "down.on_pressed", "right.on_pressed", etc.
// For complex conditions with +, |, !, it will fall back to joystick check
inline bool check_keyboard_condition(const std::string& condition, std::shared_ptr<Keyboard> keyboard) {
    if (!keyboard) return false;
    
    std::string lower_condition = condition;
    std::transform(lower_condition.begin(), lower_condition.end(), lower_condition.begin(), ::tolower);
    
    // For simple conditions (single key), check keyboard directly
    // Check for directional keys with .on_pressed
    if (lower_condition == "up.on_pressed" && keyboard->key() == "up" && keyboard->on_pressed) {
        return true;
    }
    if (lower_condition == "down.on_pressed" && keyboard->key() == "down" && keyboard->on_pressed) {
        return true;
    }
    if (lower_condition == "left.on_pressed" && keyboard->key() == "left" && keyboard->on_pressed) {
        return true;
    }
    if (lower_condition == "right.on_pressed" && keyboard->key() == "right" && keyboard->on_pressed) {
        return true;
    }
    
    // Check for directional keys with .on_released
    if (lower_condition == "up.on_released" && keyboard->key() == "up" && keyboard->on_released) {
        return true;
    }
    if (lower_condition == "down.on_released" && keyboard->key() == "down" && keyboard->on_released) {
        return true;
    }
    if (lower_condition == "left.on_released" && keyboard->key() == "left" && keyboard->on_released) {
        return true;
    }
    if (lower_condition == "right.on_released" && keyboard->key() == "right" && keyboard->on_released) {
        return true;
    }
    
    // Check for directional keys with .pressed (without on_)
    if (lower_condition == "up.pressed" && keyboard->key() == "up") {
        return true;
    }
    if (lower_condition == "down.pressed" && keyboard->key() == "down") {
        return true;
    }
    if (lower_condition == "left.pressed" && keyboard->key() == "left") {
        return true;
    }
    if (lower_condition == "right.pressed" && keyboard->key() == "right") {
        return true;
    }
    
    // For complex conditions, return false to fall back to joystick check
    return false;
}

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                
                // Check if condition uses keyboard keys
                bool uses_keyboard = condition_uses_keyboard_keys(condition);
                
                if (uses_keyboard) {
                    // For keyboard-based conditions, check keyboard first, then fall back to joystick
                    unitree::common::dsl::Parser p(condition);
                    auto ast = p.Parse();
                    auto func = unitree::common::dsl::Compile(*ast);
                    registered_checks.emplace_back(
                        std::make_pair(
                            [func, condition]()->bool{ 
                                // Check keyboard first for simple directional key conditions
                                if (check_keyboard_condition(condition, FSMState::keyboard)) {
                                    return true;
                                }
                                // Fall back to joystick check (handles complex conditions and joystick keys)
                                return func(FSMState::lowstate->joystick); 
                            },
                            fsm_id
                        )
                    );
                } else {
                    // For joystick-only conditions, use standard joystick check
                    unitree::common::dsl::Parser p(condition);
                    auto ast = p.Parse();
                    auto func = unitree::common::dsl::Compile(*ast);
                    registered_checks.emplace_back(
                        std::make_pair(
                            [func]()->bool{ return func(FSMState::lowstate->joystick); },
                            fsm_id
                        )
                    );
                }
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};