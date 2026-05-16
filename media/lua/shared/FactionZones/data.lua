local FactionZones = {}

FactionZones.zones = {
    -- Example Faction Base: Bandit Camp
    {
        name = "Bandit Camp",
        x = 10000, -- Example X coordinate
        y = 10000, -- Example Y coordinate
        radius = 50, -- Example radius in tiles
        factionId = "bandits" -- Example faction ID
    },
    -- Add more faction zones here
    {
        name = "Survivor Outpost",
        x = 10500,
        y = 10500,
        radius = 75,
        factionId = "survivors"
    }
}

return FactionZones