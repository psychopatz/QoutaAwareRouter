local FactionZones = require("FactionZones/data")

local Core = {}

-- Function to check if a player is within a faction zone
function Core.isPlayerInFactionZone(player)
    local playerX = player:getX()
    local playerY = player:getY()

    for i, zone in ipairs(FactionZones.zones) do
        local dx = playerX - zone.x
        local dy = playerY - zone.y
        local distance = math.sqrt(dx * dx + dy * dy)

        if distance <= zone.radius then
            return zone -- Player is in this zone
        end
    end

    return nil -- Player is not in any faction zone
end

return Core